#! /bin/bash
#SBATCH -J cli_test_cpu
#SBATCH -o cli_test_cpu.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=4

conda activate YOUR_PYTORCH_ENV

pythpc --config YOUR_CONFIG fit 
