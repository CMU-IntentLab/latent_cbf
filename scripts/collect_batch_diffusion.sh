#!/bin/bash
# Batch script to collect diffusion trajectories for multiple checkpoints
# Usage: ./scripts/collect_batch_diffusion.sh

set -e  # Exit on any error

# Check if we're in the right directory
if [ ! -f "scripts/collect_trajs.py" ]; then
    echo "❌ Error: scripts/collect_trajs.py not found!"
    echo "Please run this script from the latent_cbf directory."
    exit 1
fi

# Define checkpoint values: 500, 1000, 1500, ..., 10000
checkpoints=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000 10500 11000 11500 12000 12500 13000 13500 14000 14500 15000 15500 16000 16500 17000 17500 18000 18500 19000 19500)

echo "🚀 Starting batch collection for ${#checkpoints[@]} checkpoints:"
echo "Checkpoints: ${checkpoints[*]}"

successful=()
failed=()

# Run collection for each checkpoint
for checkpoint in "${checkpoints[@]}"; do
    echo ""
    echo "============================================================"
    echo "Running checkpoint $checkpoint"
    echo "============================================================"
    
    if python scripts/collect_trajs.py \
        --controller diffusion \
        --config diffusion \
        --n_trajectories 100 \
        --filename "diffusion_trajectories_${checkpoint}" \
        --output_dir /data/dubins/trajs \
        --save_images \
        --verbose \
        --checkpoint "$checkpoint"; then
        echo "✅ Checkpoint $checkpoint completed successfully!"
        successful+=("$checkpoint")
    else
        echo "❌ Checkpoint $checkpoint failed!"
        failed+=("$checkpoint")
    fi
done

# Print summary
echo ""
echo "============================================================"
echo "BATCH COLLECTION SUMMARY"
echo "============================================================"
echo "Total checkpoints: ${#checkpoints[@]}"
echo "✅ Successful (${#successful[@]}): ${successful[*]}"
echo "❌ Failed (${#failed[@]}): ${failed[*]}"

if [ ${#failed[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  Some checkpoints failed. You may want to retry them individually."
    echo "Failed checkpoint commands:"
    for checkpoint in "${failed[@]}"; do
        echo "python scripts/collect_trajs.py --controller diffusion --config diffusion --n_trajectories 50 --filename diffusion_trajectories_${checkpoint} --output_dir /data/dubins --save_images --verbose --checkpoint ${checkpoint}"
    done
    exit 1
else
    echo ""
    echo "🎉 All checkpoints completed successfully!"
    exit 0
fi

