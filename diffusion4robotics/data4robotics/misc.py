# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import functools
import os
import signal
import sys
import traceback
import yaml

import numpy as np
import torch
import wandb


GLOBAL_STEP = 0
REQUEUE_CAUGHT = False


def _signal_helper(sig, frame, prior_handler, trainer, ckpt_path):
    global REQUEUE_CAUGHT, GLOBAL_STEP
    REQUEUE_CAUGHT = True

    print(f"Caught requeue signal at step: {GLOBAL_STEP}")
    trainer.save_checkpoint(ckpt_path, GLOBAL_STEP)

    if callable(prior_handler):
        return prior_handler(sig, frame)
    return sys.exit(-1)


def set_checkpoint_handler(trainer, ckpt_path):
    global REQUEUE_CAUGHT
    REQUEUE_CAUGHT = False
    prior_handler = signal.getsignal(signal.SIGUSR2)
    handler = functools.partial(
        _signal_helper,
        prior_handler=prior_handler,
        trainer=trainer,
        ckpt_path=ckpt_path,
    )
    signal.signal(signal.SIGUSR2, handler)


def create_wandb_run(wandb_cfg, job_config, run_id=None):
    if wandb_cfg.get("debug"):
        return "null_id"
    wandb_run = wandb.init(
        project=wandb_cfg["project"],
        config=job_config,
        group=wandb_cfg.get("group"),
        name=wandb_cfg.get("name", "run"),
        id=run_id,
        resume=run_id is not None,
    )
    return wandb_run.id


def init_job(job_params, wandb_cfg, checkpoint_path, run_dir="."):
    """run_dir: directory for exp_config.yaml (same folder as checkpoint_path recommended)."""
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    exp_path = os.path.join(run_dir, "exp_config.yaml")
    params_log = {k: v for k, v in job_params.items() if k != "defaults"}
    if os.path.exists(exp_path):
        old_config = yaml.safe_load(open(exp_path, "r"))
        if os.path.exists(checkpoint_path):
            create_wandb_run(wandb_cfg, old_config["params"], run_id=old_config["wandb_id"])
            return checkpoint_path
        print(
            f"Warning: {exp_path} exists but checkpoint is missing:\n  {checkpoint_path}\n"
            "Removing stale exp_config.yaml and starting a new run."
        )
        os.remove(exp_path)
    wandb_id = create_wandb_run(wandb_cfg, params_log, run_id=None)
    save_dict = dict(wandb_id=wandb_id, params=params_log)
    with open(exp_path, "w") as f:
        yaml.dump(save_dict, f, default_flow_style=False)
    print("Training w/ Config:")
    print(yaml.dump(params_log))
    return None
