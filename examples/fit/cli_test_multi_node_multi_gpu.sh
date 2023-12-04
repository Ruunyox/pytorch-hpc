#! /bin/bash
#SBATCH -J pyt_cli_test_multi_node_multi_gpu
#SBATCH -o ./fashion_mnist_multi_node_multi_gpu/cli_test_multi_node_multi_gpu.out
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100
#SBATCH --reservation=a100_tests
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=4

module load cuda/11.8
module load anaconda3/2023.09 

conda activate base
tensorboard_dir=fashion_mnist_multi_node_multi_gpu/tensorboard

srun pythpc --config fashion_mnist_fcc_multi_node_multi_gpu.yaml fit --trainer.profiler=lightning.pytorch.profilers.AdvancedProfiler --trainer.profiler.dirpath="${tensorboard_dir}" --trainer.profiler.filename="prof"
