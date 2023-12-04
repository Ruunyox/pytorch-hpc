#! /bin/bash
#SBATCH --job-name pyt_cli_test_cpu
#SBATCH -o ./fashion_mnist/cli_test_cpu.out
#SBATCH -t 00:30:00
#SBATCH -p standard96:test
#SBATCH -N 1
#SBATCH --mem-per-cpu 1G
#SBATCH --cpus-per-task 4

module load anaconda3/2023.09

conda activate base
tensorboard_dir=fashion_mnist/tensorboard

srun pythpc --config fashion_mnist_fcc.yaml fit --trainer.profiler=lightning.pytorch.profilers.AdvancedProfiler --trainer.profiler.dirpath="${tensorboard_dir}" --trainer.profiler.filename="prof"
