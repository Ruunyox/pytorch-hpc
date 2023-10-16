#! /bin/bash
#SBATCH -J cli_test
#SBATCH -o cli_test.out
#SBATCH --time=00:10:00
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=5G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu


module load gcc/11.3.0
module load sw.a100
module load cuda/11.8

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/bzfbnick/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/bzfbnick/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/bzfbnick/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/bzfbnick/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate pytorch_test

python cli.py --config /scratch/usr/bzfbnick/pytorch_benchmarking/pytorch-hpc/examples/fit/fashion_mnist_fcc.yaml fit
