#! /bin/bash

# CPU 
sbatch cli_test_cpu.sh
sbatch cli_test_conv_cpu.sh
sbatch cli_test_graph_cpu.sh

# GPU
sbatch cli_test_gpu.sh
sbatch cli_test_conv_gpu.sh
sbatch cli_test_graph_gpu.sh

# Multi GPU
sbatch cli_test_conv_multi_gpu.sh
sbatch cli_test_graph_multi_gpu.sh

# Multnode
sbatch cli_test_multi_node_multi_gpu.sh

