#! /bin/bash
#SBATCH -J pyt_cli_test_graph_cpu
#SBATCH -o aqsol/cli_test_graph_cpu
#SBATCH --time=00:30:00
#SBATCH --partition=standard96:test
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=4

module load anaconda3/2023.09

conda activate base
tensorboard_dir=aqsol/tensorboard

pythpc --config aqsol_gcr.yaml fit --trainer.profiler=lightning.pytorch.profilers.AdvancedProfiler --trainer.profiler.dirpath="${tensorboard_dir}" --trainer.profiler.filename="prof"
