# Distributed Training with PyTorch

A practical project demonstrating distributed deep learning training using PyTorch Distributed Data Parallel (DDP) on CPU and GPU architectures.

The objective of this project is to understand how distributed training works in practice, from a simple single-process training setup to multi-process CPU and multi-GPU training.

## Features

* PyTorch model training on MNIST
* Single-process training baseline
* CPU distributed training with `torchrun` and Gloo backend
* Multi-GPU training with NCCL backend
* Automatic experiment logging with TensorBoard
* Checkpoint saving
* Performance monitoring
* Scaling analysis (speedup and efficiency)

## Project structure

```
distributed_training/
│
├── scripts 
|   ├── train.py                  # Main training script
|   ├── model.py                  # Neural network architecture
|   ├── dataset.py                # Dataset and DataLoader management
|   ├── engine.py                 # Training and evaluation loops
|   ├── distributed.py            # Distributed initialization utilities
|   ├── monitor.py                # Performance monitoring tools
|   └── utils.py                  # Utility functions
│
├── configs/
│   └── config.json               # Training configuration
│
├── test_cpu.sh                   # CPU SLURM job
├── test_gpu.sh                   # GPU SLURM job
│
├── data/                         # Dataset storage
├── checkpoints/                  # Saved models
├── runs/                         # TensorBoard logs
├── env/                          # Conda environment file
└── logs/                         # SLURM outputs
```
**NB**: **data**, **runs**, and **checkpoints** folders are generated through running
except **logs** that must be created before launching SLURM jobs.

## Environment setup

Create and activate the conda environment:

```bash
conda create -n distributed-training python=3.12
conda activate distributed-training
```

Install dependencies:

```bash
pip install -r env/requirements.txt
```

## Running locally

### Single-process training

A standard training run can be launched with:

```bash
python scripts/train.py
```

The model automatically uses:

* CPU if no GPU is available
* CUDA GPU if available

## Distributed CPU training

CPU distributed training uses PyTorch Distributed Data Parallel with the Gloo backend.

Example with 4 processes:

```bash
torchrun --standalone --nproc_per_node=4 train.py
```

Each process receives a copy of the model and trains on a subset of the dataset. Gradients are synchronized after each backward pass.

## Distributed GPU training

GPU training uses the NCCL communication backend.

Example with 2 GPUs:

```bash
torchrun --standalone --nproc_per_node=2 train.py
```

Each process is associated with one GPU.

## SLURM usage

Example GPU job:

```bash
sbatch test_gpu.sh
```

The SLURM scripts handle:

* resource allocation
* environment activation
* GPU verification
* training execution
* log redirection

Job outputs are stored in:

```
logs/
```

## TensorBoard monitoring

Training metrics are logged with TensorBoard.

Launch TensorBoard:

```bash
tensorboard --logdir runs/
```

Available metrics include:

* Training loss
* Validation loss
* Validation accuracy
* Epoch duration
* CPU usage
* RAM usage
* GPU memory usage
* Training throughput (number of images processed per second)
* Training duration

## Distributed training results

Experiments were conducted on CPU and GPU resources using PyTorch DDP.

Metrics:

* Total training time
* Throughput (images/s)
* Speedup
* Parallel efficiency

| Configuration     | Backend | Time (s) | Throughput (img/s) | Speedup | Efficiency |
| ----------------- | ------- | -------: | -----------------: | ------: | ---------: |
| CPU (1 process)   | --      |    256.9 |                268 |    1.00 |       100% |
| CPU (2 processes) | Gloo    |    106.1 |                561 |    2.42 |       121% |
| CPU (4 processes) | Gloo    |     63.2 |                905 |    4.06 |     101.5% |
| GPU (1 GPU)       | NCCL    |     58.9 |               1062 |    1.00 |       100% |
| GPU (2 GPUs)      | NCCL    |     42.0 |               1453 |    1.40 |        70% |

## Interpretation

The experiments show that:

* Distributed training significantly reduces training time.
* CPU scaling provides substantial speedup when increasing the number of processes.
* GPU training provides better absolute performance than CPU training.
* Multi-GPU training introduces communication overhead due to gradient synchronization.
* Scaling efficiency decreases as communication becomes a larger fraction of the total computation.

## Future improvements

Possible extensions:

* Multi-node distributed training
* Larger neural network architectures
* Larger datasets
* More extensive benchmarking
* Hyperparameter optimization

## Author

**Salim ISSA**

PhD Student in AI Speech processing <br>
Avignon Université – Laboratoire Informatique d'Avignon (LIA)

Email: salimissa202@gmail.com