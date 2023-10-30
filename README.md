# PyTorch-HPC

============================

### Pytorch testing suite for HPC environment/module development
-------------------------------

Simple tests to check that pytorch utilities work for HPC deployment. For rapid
testing, model building/training is controlled through `pytorch_lightning` YAML
configuration files.

### Usage
-------------------------------

Any `TorchVision` builtin dataset can be used with the configuration YAML.
Different example YAMLs and SLURM submission scripts for CPU and DDP-GPU training are included in `examples`.

After installation (`pip install .`), run the following:

`pythpc --config PATH_TO_YOUR_CONFIG fit`

To train a model. Example configs and SLURM submission scripts can be found in
the `examples` folder. Be aware that you may need to predownload/cache
datasets using login nodes if cluster nodes have no active
internet connection. 
