## Installation

This implementation requires Python 3.8 or higher. We recommend creating a new conda environment for installation.

``` sh
conda create -n trace python=3.8 -y
conda activate trace
cd trace
pip install -e .
```

### Data Preparation

To prepare the dataset for training:

```sh
cd data
python preprocess.py $dataset_name
```

### Training and Evaluation

The configuration files for different experiments are located in the `/config` directory. To start training with a specific configuration:

``` sh
python -m kge start config/icews14-best.yaml
```

The training process will automatically use all available GPUs by default. The evaluation results will be displayed in the log file.

### Resume Training and Testing

To resume training from a checkpoint or evaluate a trained model:

```sh
python -m kge resume ./local/experiments/...
```

The evaluation metrics will be shown in the log file, including MRR, Hits@1, Hits@3, and Hits@10.

# Multi-GPU training configuration(put this in yaml file)
job.multi_gpu: true

job.device: cuda
