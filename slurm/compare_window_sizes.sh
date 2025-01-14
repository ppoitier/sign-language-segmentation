#!/bin/bash
# Submission script for Lucia
#SBATCH --job-name=sls_dgs_win
#SBATCH --time=06:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=gpu
#SBATCH --array=0-10
#
#SBATCH --mail-user=pierre.poitier@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lsfb
#
#SBATCH --output=./out/win/dgs_%A_%a.out

module purge
module load EasyBuild/2022a
module load Python/3.10.4-GCCcore-11.3.0

# Activate Python virtual env
source /gpfs/home/acad/unamur-fac_info/ppoitier/envs/dl/bin/activate

config_files=(
  "../config/windows/DGS_O_MSTCN_H64_IO_W250.yaml"
  "../config/windows/DGS_O_MSTCN_H64_IO_W500.yaml"
  "../config/windows/DGS_O_MSTCN_H64_IO_W1000.yaml"
  "../config/windows/DGS_O_MSTCN_H64_IO_W1500.yaml"
  "../config/windows/DGS_O_MSTCN_H64_IO_W2000.yaml"
  "../config/windows/DGS_O_MSTCN_H64_IO_W2500.yaml"
  "../config/windows/DGS_O_MSTCN_H64_IO_W3000.yaml"
  "../config/windows/DGS_O_MSTCN_H64_IO_W3500.yaml"
  "../config/windows/DGS_O_MSTCN_H64_IO_W4000.yaml"
  "../config/windows/DGS_O_MSTCN_H64_IO_W4500.yaml"
  "../config/windows/DGS_O_MSTCN_H64_IO_W5000.yaml"
)

config_file=${config_files[$SLURM_ARRAY_TASK_ID]}

nvidia-smi
echo "Job array ID: $SLURM_ARRAY_TASK_ID"
echo "Job start at $(date)"
python ../main.py --config-path="$config_file"
echo "Job end at $(date)"