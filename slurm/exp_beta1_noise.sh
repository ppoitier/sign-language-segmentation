#!/bin/bash
# Submission script for Lucia
#SBATCH --job-name=sls_beta1_noise
#SBATCH --time=08:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=gpu
#SBATCH --array=0-0
#
#SBATCH --mail-user=pierre.poitier@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lsfb
#
#SBATCH --output=./out/beta1/noise/%A_%a.out

module purge
module load EasyBuild/2022a
module load Python/3.10.4-GCCcore-11.3.0

# Activate Python virtual env
source /gpfs/home/acad/unamur-fac_info/ppoitier/envs/dl/bin/activate

config_files=(
#  "../config/beta1/noise/low/dgs_io_noise_low.yaml"
#  "../config/beta1/noise/low/dgs_io_off_noise_low.yaml"
#  "../config/beta1/noise/low/lsfb_io_noise_low.yaml"
#  "../config/beta1/noise/low/lsfb_io_off_noise_low.yaml"
#  "../config/beta1/noise/medium/dgs_io_noise_medium.yaml"
#  "../config/beta1/noise/medium/dgs_io_off_noise_medium.yaml"
#  "../config/beta1/noise/medium/lsfb_io_noise_medium.yaml"
#  "../config/beta1/noise/medium/lsfb_io_off_noise_medium.yaml"
#  "../config/beta1/noise/high/dgs_io_noise_high.yaml"
#  "../config/beta1/noise/high/dgs_io_off_noise_high.yaml"
#  "../config/beta1/noise/high/lsfb_io_noise_high.yaml"
#  "../config/beta1/noise/high/lsfb_io_off_noise_high.yaml"
#  "../config/beta1/noise/remove-gaps/dgs_io_no_gap.yaml"
#  "../config/beta1/noise/remove-gaps/dgs_io_off_no_gap.yaml"
#  "../config/beta1/noise/remove-gaps/lsfb_io_no_gap.yaml"
  "../config/beta1/noise/remove-gaps/lsfb_io_off_no_gap.yaml"
)

config_file=${config_files[$SLURM_ARRAY_TASK_ID]}

nvidia-smi
echo "Job array ID: $SLURM_ARRAY_TASK_ID"
echo "Job start at $(date)"
python ../main.py --config-path="$config_file"
echo "Job end at $(date)"