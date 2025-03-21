#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 6:00:00
#SBATCH --gpus=h100-80:1
#SBATCH --array=0-31  # 32 tasks (0 to 31)
#SBATCH -o output_%A_%a.out  # Output file per task (%A=job ID, %a=array index)
#SBATCH -e error_%A_%a.err   # Error file per task

module load anaconda3
conda activate env_214

# Define parameter combinations
LAYER_COMBOS=("1 1 1" "1 1 2" "1 2 1" "1 2 2" "2 1 1" "2 1 2" "2 2 1" "2 2 2")
AUG_FLAGS=("" "--augmentation-flip" "--augmentation-rotate" "--augmentation-flip --augmentation-rotate")

# Calculate indices from SLURM_ARRAY_TASK_ID (0-31)
LAYER_IDX=$((SLURM_ARRAY_TASK_ID / 4))  # 0-7 (8 layer combos)
AUG_IDX=$((SLURM_ARRAY_TASK_ID % 4))    # 0-3 (4 aug combos)

# Extract parameters for this task
LAYERS="${LAYER_COMBOS[$LAYER_IDX]}"
AUG="${AUG_FLAGS[$AUG_IDX]}"

# Run the Python script with the selected parameters
python "/jet/home/azhang19/stat 214/stat-214-lab2-group6/code/modeling/train_autoencoder.py" --num-layers-block $LAYERS $AUG

echo "Task $SLURM_ARRAY_TASK_ID: Layers=$LAYERS, Aug=$AUG"