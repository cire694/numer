#!/bin/bash
#SBATCH --job-name=eval_ensemble                    # name shown in squeue
#SBATCH --output=logs/%j_eval_ensemble.out          # stdout log (%j = job id)
#SBATCH --error=logs/%j_eval_ensemble.err           # stderr log
#SBATCH --time=08:00:00                             # max wall time (HH:MM:SS)
#SBATCH --nodes=1                                   # single node (joblib doesn't span nodes)
#SBATCH --ntasks=1                                  # one task (our python script)
#SBATCH --cpus-per-task=16                          # cores for joblib parallelism
#SBATCH --mem=128G                                  # RAM — Numerai data is large
#SBATCH --partition=mit_normal                      # ORCD partition name (check with sinfo)

# ── Environment setup ────────────────────────────────────────────
echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

# Load conda — path may differ, check with: which conda
module load miniforge
source /home/$USER/.bashrc

# Use the conda env interpreter directly so Slurm jobs do not rely on PATH
PYTHON_BIN="${PYTHON_BIN:-/home/$USER/.conda/envs/numer/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Error: Python not found at $PYTHON_BIN" >&2
    exit 1
fi

# Move to project directory
cd /home/$USER/numer

# Create logs dir if it doesn't exist
mkdir -p logs

# ── Run training ─────────────────────────────────────────────────
"$PYTHON_BIN" -m evaluate_models.eval_ensemble

echo "Job finished: $(date)"