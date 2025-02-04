#!/bin/bash
# Submission script for Lucia
#SBATCH --job-name=sls_dgs_improv1
#SBATCH --time=08:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=gpu
#SBATCH --array=0-3
#
#SBATCH --mail-user=pierre.poitier@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lsfb
#
#SBATCH --output=./out/improv2/dgs_%A_%a.out

module purge
module load EasyBuild/2022a
module load Python/3.10.4-GCCcore-11.3.0

# Activate Python virtual env
source /gpfs/home/acad/unamur-fac_info/ppoitier/envs/dl/bin/activate

config_files=(
  "../config/improv1/dgs_pn_mstcn_4s_10l_io_ce.yaml"
  "../config/improv1/dgs_pn_mstcn_4s_10l_io_fl.yaml"
  "../config/improv1/dgs_pn_mstcn_4s_10l_io_ce_off.yaml"
  "../config/improv1/dgs_pn_mstcn_4s_10l_io_fl_off.yaml"
)

config_file=${config_files[$SLURM_ARRAY_TASK_ID]}

nvidia-smi
echo "Job array ID: $SLURM_ARRAY_TASK_ID"
echo "Job start at $(date)"
python ../main.py --config-path="$config_file"
echo "Job end at $(date)"