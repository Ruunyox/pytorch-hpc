#! /bin/bash
#SBATCH -J cli_test_multi_node_multi_gpu
#SBATCH -o cli_test_multi_node_multi_gpu.out
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=4

conda activate YOUR_PYTORCH_ENV

srun pythpc --config YOUR_CONFIG fit
