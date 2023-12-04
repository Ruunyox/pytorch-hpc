#! /bin/bash
#SBATCH -J pyt_cli_test_graph_gpu
#SBATCH -o aqsol_gpu/cli_test_graph_gpu
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100
#SBATCH --reservation=a100_tests
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=4

module load cuda/11.8
module load anaconda3/2023.09 

conda activate base
tensorboard_dir=aqsol_gpu/tensorboard

srun pythpc --config aqsol_gcr_gpu.yaml fit --trainer.profiler=lightning.pytorch.profilers.AdvancedProfiler --trainer.profiler.dirpath="${tensorboard_dir}" --trainer.profiler.filename="prof"
