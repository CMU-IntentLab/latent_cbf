# Copyright (c) Sudeep Dasari, 2023

"""BC finetune: dubins (obs_dim=1, ac_dim=1) + DiffusionUnet + RobomimicResNet."""

import argparse
import os
import sys
import traceback
from pathlib import Path

# Repo `src/` so `latent_cbf.configs.paths` matches the rest of the pipeline.
_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if _SRC_ROOT.is_dir() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from latent_cbf.configs.paths import (  # noqa: E402
    DIFFUSION4ROBOTICS_DEFAULTS,
    DIFFUSION_DIR,
)

import numpy as np
import torch
import tqdm
import yaml

from data4robotics import misc, transforms
from data4robotics.models.diffusion_unet import DiffusionUnetAgent
from data4robotics.models.resnet import RobomimicResNet
from data4robotics.replay_buffer import RobobufReplayBuffer
from data4robotics.task import BCTask
from data4robotics.trainers.bc import BehaviorCloning
from data4robotics.trainers.utils import optim_builder, schedule_builder

_DEFAULTS_PATH = str(DIFFUSION4ROBOTICS_DEFAULTS)


def _load_defaults(path=None):
    path = path or _DEFAULTS_PATH
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_args():
    p = argparse.ArgumentParser(description="Dubins diffusion BC finetune")
    p.add_argument("--buffer-path", type=str, required=True)
    p.add_argument("--cam-indexes", type=int, nargs="*", default=[0])
    p.add_argument("--exp-name", type=str, default="dubins_diffusion")
    p.add_argument("--wandb-name", type=str, default="dubins_run")
    p.add_argument("--max-iterations", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=10)
    p.add_argument("--seed", type=int, default=292285)
    p.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Full path to .ckpt, or relative name under checkpoint-dir (see defaults.yaml)",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override defaults.yaml checkpoint_dir for this run",
    )
    p.add_argument(
        "--train-transform",
        type=str,
        default="medium",
        choices=["preproc", "basic", "medium", "hard", "advanced", "gpu_medium"],
    )
    p.add_argument("--ac-chunk", type=int, default=16)
    p.add_argument("--img-chunk", type=int, default=1)
    p.add_argument("--feature-dim", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--eval-freq", type=int, default=500)
    p.add_argument("--save-freq", type=int, default=500)
    p.add_argument("--defaults", type=str, default=None, help="Override defaults.yaml path")
    return p.parse_args()


def _device_id(device_str: str):
    """BaseTrainer uses tensor.to(device_id); accept cuda:N or cpu."""
    if device_str.startswith("cuda"):
        return torch.device(device_str)
    return torch.device(device_str)


def _resolve_checkpoint_path(args, defaults):
    """checkpoint_dir from --checkpoint-dir, else defaults.yaml, else ``DIFFUSION_DIR`` from ``latent_cbf.configs.paths``."""
    ckpt_dir = args.checkpoint_dir
    if ckpt_dir is None:
        ckpt_dir = defaults.get("checkpoint_dir")
    if ckpt_dir is None:
        ckpt_dir = str(DIFFUSION_DIR)
    ckpt_dir = os.path.expanduser(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    if args.checkpoint_path:
        p = os.path.expanduser(args.checkpoint_path)
        if os.path.isabs(p):
            d = os.path.dirname(p)
            os.makedirs(d, exist_ok=True)
            # exp_config / agent live next to the checkpoint file
            return d, p
        return ckpt_dir, os.path.join(ckpt_dir, p)
    return ckpt_dir, os.path.join(ckpt_dir, f"{args.exp_name}_latest.ckpt")


def main():
    args = _parse_args()
    defaults = _load_defaults(args.defaults)
    run_dir, checkpoint_path = _resolve_checkpoint_path(args, defaults)

    job_params = {
        "buffer_path": args.buffer_path,
        "cam_indexes": args.cam_indexes,
        "exp_name": args.exp_name,
        "max_iterations": args.max_iterations,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "train_transform": args.train_transform,
        "ac_chunk": args.ac_chunk,
        "img_chunk": args.img_chunk,
        "feature_dim": args.feature_dim,
        "device": args.device,
        "eval_freq": args.eval_freq,
        "save_freq": args.save_freq,
        "checkpoint_dir": run_dir,
        "checkpoint_path": checkpoint_path,
        "defaults": defaults,
    }
    wandb_cfg = dict(
        name=args.wandb_name,
        project=defaults["wandb"]["project"],
        group=args.exp_name,
        debug=defaults["wandb"]["debug"],
    )

    try:
        resume_model = misc.init_job(job_params, wandb_cfg, checkpoint_path, run_dir=run_dir)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed + 1)

        r = defaults["resnet"]
        features = RobomimicResNet(
            size=r["size"],
            norm_cfg=r["norm_cfg"],
            weights=r["weights"],
            img_size=r["img_size"],
            feature_dim=args.feature_dim,
        )
        n_cams = len(args.cam_indexes)
        agent = DiffusionUnetAgent(
            features=features,
            shared_mlp=[],
            odim=1,
            n_cams=n_cams,
            use_obs=False,
            ac_dim=1,
            ac_chunk=args.ac_chunk,
            train_diffusion_steps=defaults["train_diffusion_steps"],
            eval_diffusion_steps=defaults["eval_diffusion_steps"],
            imgs_per_cam=args.img_chunk,
            dropout=defaults["dropout"],
            share_cam_features=False,
            feat_batch_norm=False,
            noise_net_kwargs=defaults["noise_net_kwargs"],
        )

        agent_yaml_path = os.path.join(run_dir, "agent_config.yaml")
        with open(agent_yaml_path, "w") as f:
            yaml.dump(
                {
                    "features": {
                        "class": "RobomimicResNet",
                        "feature_dim": args.feature_dim,
                        **r,
                    },
                    "agent": "DiffusionUnetAgent",
                    "odim": 1,
                    "ac_dim": 1,
                    "n_cams": n_cams,
                    "noise_net_kwargs": defaults["noise_net_kwargs"],
                },
                f,
            )

        o = defaults["optim"]
        ob = optim_builder("AdamW", {"lr": args.lr, **o})
        sched_kwargs = {
            "num_warmup_steps": defaults["schedule"]["num_warmup_steps"],
            "num_training_steps": args.max_iterations,
        }
        sb = schedule_builder("cosine", sched_kwargs, from_diffusers=True)
        device = _device_id(args.device)
        trainer = BehaviorCloning(
            model=agent, device_id=device, optim_builder=ob, schedule_builder=sb
        )

        train_tf = transforms.get_transform_by_name(args.train_transform)
        past_frames = args.img_chunk - 1
        train_buffer = RobobufReplayBuffer(
            buffer_path=args.buffer_path,
            transform=train_tf,
            n_test_trans=defaults["n_test_trans"],
            ac_chunk=args.ac_chunk,
            mode="train",
            cam_indexes=args.cam_indexes,
            past_frames=past_frames,
            ac_dim=1,
            artifact_dir=run_dir,
        )
        test_buffer = RobobufReplayBuffer(
            buffer_path=args.buffer_path,
            transform=transforms.get_transform_by_name("preproc"),
            n_test_trans=defaults["n_test_trans"],
            ac_chunk=args.ac_chunk,
            mode="test",
            cam_indexes=args.cam_indexes,
            past_frames=past_frames,
            ac_dim=1,
            artifact_dir=run_dir,
        )
        task = BCTask(
            train_buffer,
            test_buffer,
            n_cams=n_cams,
            obs_dim=1,
            ac_dim=1,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        gpu_transform = (
            transforms.get_gpu_transform_by_name(args.train_transform)
            if "gpu" in args.train_transform
            else None
        )

        print(resume_model)
        if resume_model is not None:
            misc.GLOBAL_STEP = trainer.load_checkpoint(resume_model)
        elif misc.GLOBAL_STEP == 0:
            trainer.save_checkpoint(checkpoint_path, misc.GLOBAL_STEP)
        assert misc.GLOBAL_STEP >= 0, "GLOBAL_STEP not loaded correctly!"

        misc.set_checkpoint_handler(trainer, checkpoint_path)
        print(f"Checkpoints under {run_dir}")
        print(f"Starting at Global Step {misc.GLOBAL_STEP}")

        trainer.set_train()
        train_iterator = iter(task.train_loader)
        sf = defaults["schedule_freq"]
        for itr in (
            pbar := tqdm.tqdm(range(args.max_iterations), postfix=dict(Loss=None))
        ):
            if itr < misc.GLOBAL_STEP:
                continue
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(task.train_loader)
                batch = next(train_iterator)

            if gpu_transform is not None:
                (imgs, obs), actions, mask = batch
                imgs = {k: v.to(trainer.device_id) for k, v in imgs.items()}
                imgs = {k: gpu_transform(v) for k, v in imgs.items()}
                batch = ((imgs, obs), actions, mask)

            trainer.optim.zero_grad()
            loss = trainer.training_step(batch, misc.GLOBAL_STEP)
            loss.backward()
            trainer.optim.step()

            pbar.set_postfix(dict(Loss=loss.item()))
            misc.GLOBAL_STEP += 1

            if misc.GLOBAL_STEP % sf == 0:
                trainer.step_schedule()

            if misc.GLOBAL_STEP % args.eval_freq == 0:
                trainer.set_eval()
                task.eval(trainer, misc.GLOBAL_STEP)
                trainer.set_train()

            if misc.GLOBAL_STEP >= args.max_iterations:
                trainer.save_checkpoint(checkpoint_path, misc.GLOBAL_STEP)
                return
            elif misc.GLOBAL_STEP % args.save_freq == 0:
                trainer.save_checkpoint(checkpoint_path, misc.GLOBAL_STEP)
                stem = checkpoint_path.rsplit(".", 1)[0]
                save_path = f"{stem}{misc.GLOBAL_STEP}.ckpt"
                trainer.save_checkpoint(save_path, misc.GLOBAL_STEP)
    except Exception:
        err_path = os.path.join(run_dir, "exception.log")
        traceback.print_exc(file=open(err_path, "w"))
        with open(err_path, "r") as f:
            print(f.read())


if __name__ == "__main__":
    main()
