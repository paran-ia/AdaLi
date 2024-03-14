import torch
import torch.nn as nn
import argparse
from myutils.built_data import build_cifar,bulid_cifar10_dvs,build_dvs_gesture
from models.resnet import resnet18_cifar
from models.vgg import vgg11,vgg16
from modules.neuron import LIF,IF
from arguments import arg
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name',
                        choices=['CIFAR10DVS', 'CIFAR10', 'CIFAR100','DVSGesture'])
    parser.add_argument('--arch', default='resnet18', type=str, help='arch name',
                        choices=['resnet18','vgg11'])
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--learning_rate', default=1e-1, type=float, help='initial learning_rate')
    parser.add_argument('--epochs', default=500, type=int, help='number of training epochs')
    parser.add_argument('--neuron', default='LIF', type=str, help='snn neuron',
                        choices=['LIF', 'IF'])
    parser.add_argument('--step', default=4, type=int, help='snn step')
    parser.add_argument('--reset', default=True, type=bool, help='reset to 0')
    parser.add_argument('--Vth', default=1.0, type=float, help='threshold')
    parser.add_argument('--wl', default=0.5, type=float, help='left width')
    parser.add_argument('--wr', default=0.5, type=float, help='right width')
    parser.add_argument('--adaptive', default="vanilla",type=str, help="adaptive function",
                        choices=['log','linear','sqrt'])
    args = parser.parse_args()
    arg['Epoch'] = args.epochs
    model_save_name = f'raw/{args.dataset}/{args.dataset}_{args.arch}_{args.neuron}_T{args.step}_{args.adaptive}_wl{args.wl}_wr{args.wr}_b{args.batch_size}_e{args.epochs}.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.neuron == 'LIF':
        neuron = LIF
    else:
        neuron = IF

    input_channels = 2

    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        input_channels = 3
        use_cifar10 = args.dataset == 'CIFAR10'
        if use_cifar10:
            num_classes = 10
        else:
            num_classes = 100
        train_loader, test_loader = build_cifar(cutout=True, use_cifar10=use_cifar10, auto_aug=True,batch_size=args.batch_size)
    elif args.dataset == 'CIFAR10DVS':
        num_classes = 10
        train_loader, test_loader = bulid_cifar10_dvs(args.step,args.batch_size,num_workers=0)
    elif args.dataset == 'DVSGesture':
        num_classes = 11
        train_loader, test_loader = build_dvs_gesture(args.step,args.batch_size,num_workers=0)
    else:
        raise NotImplementedError


    best_acc = 0
    best_epoch = 0
    if args.arch == 'resnet18':
        snn = resnet18_cifar(input_c=input_channels, num_classes=num_classes, neuron=neuron, Vth=args.Vth,
                             step=args.step, wl=args.wl, wr=args.wr, adaptive=args.adaptive, reset=args.reset)
    elif args.arch == 'vgg11':
        snn = vgg11(input_c=input_channels, num_classes=num_classes, neuron=neuron, neuron_dropout=0.0, Vth=args.Vth,
                    step=args.step, wl=args.wl, wr=args.wr, adaptive=args.adaptive, reset=args.reset)
    elif args.arch == 'vgg16':
        snn = vgg16(input_c=input_channels, num_classes=num_classes, neuron=neuron, neuron_dropout=0.0, Vth=args.Vth,
                    step=args.step, wl=args.wl, wr=args.wr, adaptive=args.adaptive, reset=args.reset)
    else:
        raise NotImplementedError

    snn.to(device)
    snn.load_state_dict(torch.load(model_save_name))
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(params=snn.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    snn.eval()

    correct = 0
    total = 0
    acc = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
                inputs = inputs.repeat(args.step, 1, 1, 1, 1)
            else:
                inputs = inputs.transpose(0, 1)
            outputs = snn(inputs)
            _, predicted = outputs.cpu().max(1)
            total += (targets.size(0))
            correct += (predicted.eq(targets.cpu()).sum().item())

    acc = 100 * correct / total
    print(f'Test Accuracy of the model on the test images: {acc}')
