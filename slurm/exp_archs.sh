#!/bin/bash
# Submission script for Lucia
#SBATCH --job-name=sls_dgs_archs
#SBATCH --time=06:00:00 # hh:mm:ss
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
#SBATCH --output=./out/archs/dgs_%A_%a.out

module purge
module load EasyBuild/2022a
module load Python/3.10.4-GCCcore-11.3.0

# Activate Python virtual env
source /gpfs/home/acad/unamur-fac_info/ppoitier/envs/dl/bin/activate

config_files=(
#  "../config/archs/dgs_o_lstm_h64_1l_io.yaml"
#  "../config/archs/dgs_o_lstm_h64_2l_io.yaml"
#  "../config/archs/dgs_o_lstm_h64_4l_io.yaml"
#  "../config/archs/dgs_o_lstm_h128_1l_io.yaml"
#  "../config/archs/dgs_o_lstm_h128_2l_io.yaml"
#  "../config/archs/dgs_o_lstm_h128_4l_io.yaml"
#  "../config/archs/dgs_o_lstm_h256_1l_io.yaml"
#  "../config/archs/dgs_o_lstm_h256_2l_io.yaml"
#  "../config/archs/dgs_o_lstm_h256_4l_io.yaml"
#  "../config/archs/dgs_o_lstm_h512_1l_io.yaml"
#  "../config/archs/dgs_o_lstm_h512_2l_io.yaml"
#  "../config/archs/dgs_o_lstm_h512_4l_io.yaml"
  "../config/archs/dgs_o_mstcn_h64_4s_6l_io.yaml"
  "../config/archs/dgs_o_mstcn_h64_4s_10l_io.yaml"
  "../config/archs/dgs_o_mstcn_h128_4s_6l_io.yaml"
  "../config/archs/dgs_o_mstcn_h128_4s_10l_io.yaml"
)

config_file=${config_files[$SLURM_ARRAY_TASK_ID]}

nvidia-smi
echo "Job array ID: $SLURM_ARRAY_TASK_ID"
echo "Job start at $(date)"
python ../main.py --config-path="$config_file"
echo "Job end at $(date)"