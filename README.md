# AdaLi

This repository is the official PyTorch implementation of the paper: Towards Adaptive and Lightweight Backpropagation for Training Spiking Neural Networks.

# Dependencies

- Python >= 3.9
- spikingjelly==0.0.0.0.14
- torch==2.4.1+cu121
- tensorboard==2.17.1
- numpy==2.0.2

# Training

The boundary values need to be custom-modified in the config.py file. The following is an example of training with the AdaLi method. If other methods are used and no adaptive function is required, the --adaptive parameter should be removed. If training with a combined loss function, please turn on the --mix and --lamda parameters.

## CIFAR-10

```bash
python -u train.py --dataset CIFAR10 --arch resnet18_cifar --neuron IF --step 4 --epochs 500 --surrogate adali --adaptive log
```

## CIFAR-100

```bash
python -u train.py --dataset CIFAR100 --arch resnet18_cifar --neuron IF --step 4 --epochs 500 --surrogate adali --adaptive log
```

## CIFAR-10 DVS

```bash
python -u train.py --dataset CIFAR10DVS --arch vgg16 --neuron IF --step 16 --batch_size 64 --epochs 500 --surrogate adali --adaptive log
```

## DVSGesture

```bash
python -u train.py --dataset DVSGesture --arch dvsgestureNet --neuron IF --step 16 --batch_size 32 --epochs 500 --surrogate adali --adaptive log
```

# Credits

The code for the data preprocessing of DVS-CIFAR10 and DVSGesture is based on the [spikingjelly](https://github.com/fangwei123456/spikingjelly) repo.  The code for some models are from the [SLTT](https://github.com/qymeng94/SLTT) and [MPBN](https://github.com/yfguo91/MPBN) repos.

