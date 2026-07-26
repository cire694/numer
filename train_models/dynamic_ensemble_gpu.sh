#!/bin/bash
#SBATCH --job-name=dynamic_ensemble_lgbm_gpu        # name shown in squeue
#SBATCH --output=logs/%j_dynamic_ensemble_lgbm_gpu.out  # stdout log (%j = job id)
#SBATCH --error=logs/%j_dynamic_ensemble_lgbm_gpu.err   # stderr log
#SBATCH --time=06:00:00                             # max wall time (HH:MM:SS)
#SBATCH --nodes=1                                   # single node
#SBATCH --ntasks=1                                  # one task (our python script)
#SBATCH --cpus-per-task=16                          # cores for joblib parallelism
#SBATCH --mem=128G                                  # RAM — Numerai data is large
#SBATCH --partition=mit_normal_gpu                  # ORCD partition name
#SBATCH --gres=gpu:1                                # request 1 GPU

# ── Environment setup ────────────────────────────────────────────
echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "GPUS allocated: $SLURM_GPUS_ON_NODE $CUDA_VISIBLE_DEVICES"

# Load conda — path may differ, check with: which conda
module load miniforge

# Optionally load CUDA / GPU drivers if your cluster requires it
# module load cuda/11.8

# Move to project directory
cd /home/$USER/numer

# Create logs dir if it doesn't exist
mkdir -p logs

# Cap thread usage for OpenMP/BLAS inside each process
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# ── Run training on GPU ────────────────────────────────────────────
conda run -n numer python -m train_models.dynamic_ensemble

echo "Job finished: $(date)"
