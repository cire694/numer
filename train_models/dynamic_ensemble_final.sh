#!/bin/bash
#SBATCH --job-name=dynamic_ensemble_lgbm_final            # name shown in squeue
#SBATCH --output=logs/%j_dynamic_ensemble_lgbm_final.out  # stdout log (%j = job id)
#SBATCH --error=logs/%j_dynamic_ensemble_lgbm_final.err   # stderr log
#SBATCH --time=06:00:00                             # max wall time (HH:MM:SS)
#SBATCH --nodes=1                                   # single node
#SBATCH --ntasks=1                                  # one task (our python script)
#SBATCH --cpus-per-task=64                          # cores for base-model parallelism
#SBATCH --mem=256G                                  # RAM — Numerai data is large
#SBATCH --partition=mit_normal                      # CPU-only partition — no GPU is used by this pipeline

# ── Environment setup ────────────────────────────────────────────
echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

module load miniforge

cd /home/$USER/numer
mkdir -p logs

# Cap thread usage for OpenMP/BLAS inside each process
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Force Python to flush stdout immediately instead of block-buffering it —
# without this, print() output can sit in a buffer and never make it to
# the .out log if the job is killed by the time limit.
export PYTHONUNBUFFERED=1

# ── Run training ─────────────────────────────────────────────────
if [ -z "$1" ]; then
    job_id="${SLURM_JOB_ID}"
    resume_flag=""
    echo "Starting fresh training run with job ID: $job_id"
else
    job_id="$1"
    resume_flag="--resume"
    echo "Resuming from checkpoint job ID: $job_id"
fi

# n_jobs=2 lets two base models train concurrently (CPU budget is split
# between them automatically inside the script — with cpus-per-task=64
# that's ~32 threads per concurrent model). checkpoint-every-iters=250
# saves a partial booster every 250 boosting rounds within each model, so a
# timeout mid-model loses at most ~250 rounds instead of the whole model.
# --mode final routes to train_all() internally, which trains on
# train+validation combined with a 3% era early-stopping holdout (see
# --early-stopping-holdout-frac). Set it to 0 to disable early stopping and
# force every base tree to run the full n_estimators.
#
# --use-cuda is intentionally NOT passed here: the meta head is small
# enough that GPU offers no real benefit, and this job now runs on the
# GPU-free mit_normal partition
cmd="python -u -m train_models.dynamic_ensemble --mode final --job-id $job_id $resume_flag --n-jobs 2 --checkpoint-every-iters 250 --early-stopping-holdout-frac 0.03"

echo "Running: $cmd"
conda run -n numer bash -lc "$cmd"

echo "Job finished: $(date)"