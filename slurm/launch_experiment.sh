#!/bin/bash
# Submission script for Lucia
#SBATCH --job-name=sls_dgs
#SBATCH --time=24:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=gpu
#
#SBATCH --mail-user=pierre.poitier@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lsfb
#
#SBATCH --output=./dgs.out

module purge
module load EasyBuild/2022a
module load Python/3.10.4-GCCcore-11.3.0

# Activate Python virtual env
source /gpfs/home/acad/unamur-fac_info/ppoitier/envs/dl/bin/activate

config_files=(
  "../config/dgs_replication.yaml"
)

config_file=${config_files[0]}

nvidia-smi
echo "Job start at $(date)"
python ../main.py --config-path="$config_file"
echo "Job end at $(date)"