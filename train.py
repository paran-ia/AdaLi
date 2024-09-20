import argparse
from config import AdaLiConfig
import torch
import datetime
import time
import os
from modules.neuron import IF, LIF
from modules.sgSigmoid import Sigmoid
from modules.sgAdaLi import AdaLi
from modules.sgTriangle import PiecewiseQuadratic
from modules.sgSoftSign import SoftSign
from myutils.build_data import build_cifar, build_dvsgesture, build_tinyimagenet, build_cifar10dvs
from models.resnet import spiking_resnet18, spiking_resnet34
from models.resnet_imagenet import spiking_nfresnet34,spiking_nfresnet50
from models.vgg import spiking_vgg11_bn,spiking_vgg13_bn, spiking_vgg16_bn,spiking_vgg19_bn
from models.resnet_cifar import resnet18_cifar,resnet19_cifar,resnet20_cifar_modified
import torch.nn as nn
import numpy as np
from spikingjelly.activation_based import functional, monitor
from myutils.handle_config import handleConfig
from myutils.common import cal_firing_rate, cal_mp_summary_statistics
from models.dvsgesturenet import dvsgestureNet
import torch.nn.functional as F

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name',
                        choices=['CIFAR10DVS', 'CIFAR10', 'CIFAR100', 'DVSGesture', 'TinyImageNet'])
    parser.add_argument('--arch', default='resnet18_cifar', type=str, help='arch name',
                        choices=['resnet18', 'resnet34','nfresnet34', 'nfresnet50', 'vgg11', 'vgg13','vgg16','vgg19'
                            ,'dvsgestureNet','resnet18_cifar','resnet19_cifar','resnet20_cifar_modified'])
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='initial learning_rate')
    parser.add_argument('--epochs', default=500, type=int, help='number of training epochs')
    parser.add_argument('--loss', default='ce', type=str, help='loss type',
                        choices=['ce','mse','mix'])
    parser.add_argument('--lambda', default=0.05, type=float, help='weight of mse loss')
    parser.add_argument('--show_fr', action='store_true', help='store firing rate')
    parser.add_argument('--mp_statistic', action='store_true', help='store membrane potential')
    parser.add_argument('--neuron', default='IF', type=str, help='snn neuron',
                        choices=['LIF', 'IF'])
    parser.add_argument('--step', default=2, type=int, help='snn step')
    parser.add_argument('--Vth', default=1., type=float, help='threshold')
    parser.add_argument('--tau', default=1.1, type=float, help='LIF param')
    parser.add_argument('--opt', default="sgd", type=str, help="optimizer",
                        choices=['adam', 'sgd'])
    parser.add_argument('--surrogate', default="adali", type=str, help="surrogate function",
                        choices=['adali', 'sigmoid', 'triangle', 'softsign'])
    parser.add_argument('--adaptive', default="null", type=str, help="adaptive function",
                        choices=['log', 'linear', 'sqrt', 'null'])
    args = parser.parse_args()

    AdaLiConfig['Epoch'] = args.epochs
    AdaLiConfig['Vth'] = args.Vth
    data_dir = {'CIFAR10': "data/cifar10", 'CIFAR100': "data/cifar100",
                'CIFAR10DVS': "data/cifar10dvs/download", 'DVSGesture': "data/dvsgesture",
                'TinyImageNet': "data/tiny-imagenet-200"}
    curtime = datetime.datetime.now().strftime('%Y-%m-%d_%H：%M：%S')
    folder_path = [f"experiments_data/{args.dataset}/{curtime}", "data/cifar10", "data/cifar100", "data/cifar10dvs",
                   "data/dvsgesture", "data/tiny-imagenet-200"]
    for folder in folder_path:
        if not os.path.exists(folder):
            os.makedirs(folder)

    model_save_name = f'experiments_data/{args.dataset}/{curtime}/{args.dataset}_{args.arch}_{args.neuron}_T{args.step}_b{args.batch_size}_e{args.epochs}.pth'
    acc_np = []
    acc_save_path = f'experiments_data/{args.dataset}/{curtime}/{args.dataset}_{args.arch}_{args.neuron}_T{args.step}_b{args.batch_size}_e{args.epochs}.npy'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'model_save_name:{model_save_name}')
    print(f'acc_save_path:{acc_save_path}')

    if args.neuron == 'LIF':
        neuron = LIF
    else:
        neuron = IF

    if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
        input_channels = 3
        use_cifar10 = args.dataset == 'CIFAR10'
        if use_cifar10:
            num_classes = 10
        else:
            num_classes = 100
        train_loader, test_loader = build_cifar(dataset=args.dataset, data_dir=data_dir[args.dataset],
                                                batch_size=args.batch_size, num_workers=0)
    elif args.dataset == 'CIFAR10DVS':
        input_channels = 2
        num_classes = 10
        train_loader, test_loader = build_cifar10dvs(data_dir=data_dir[args.dataset], batch_size=args.batch_size,
                                                     frames_num=args.step, num_workers=0)
    elif args.dataset == 'DVSGesture':
        input_channels = 2
        num_classes = 11
        train_loader, test_loader = build_dvsgesture(data_dir=data_dir[args.dataset], batch_size=args.batch_size,
                                                     frames_num=args.step, num_workers=0)
    elif args.dataset == 'TinyImageNet':
        input_channels = 3
        num_classes = 200
        train_loader, test_loader = build_tinyimagenet(data_dir=data_dir[args.dataset], batch_size=args.batch_size,
                                                       num_workers=0)
    else:
        raise NotImplementedError(f'暂未实现该数据集{args.dataset}')

    if args.surrogate == 'adali':
        sg = AdaLi
    elif args.surrogate == 'sigmoid':
        sg = Sigmoid
    elif args.surrogate == 'triangle':
        sg = PiecewiseQuadratic
    elif args.surrogate == 'softsign':
        sg = SoftSign
    else:
        raise ValueError(f'暂未实现此代理梯度{args.surrogate}')

    if args.arch == 'resnet18':
        snn = spiking_resnet18(neuron=neuron, num_classes=num_classes,
                               input_channels=input_channels, surrogate_function=sg, v_threshold=args.Vth, tau=args.tau)
    elif args.arch == 'resnet34':
        snn = spiking_resnet34(neuron=neuron, num_classes=num_classes,
                               input_channels=input_channels, surrogate_function=sg, v_threshold=args.Vth, tau=args.tau)
    elif args.arch == 'nfresnet34':
        snn = spiking_nfresnet34(neuron=neuron, surrogate_function=sg, v_threshold=args.Vth, tau=args.tau)
    elif args.arch == 'nfresnet50':
        snn = spiking_nfresnet50(neuron=neuron, surrogate_function=sg, v_threshold=args.Vth, tau=args.tau)
    elif args.arch == 'vgg11':
        snn = spiking_vgg11_bn(neuron=neuron, num_classes=num_classes,
                               input_channels=input_channels, surrogate_function=sg, v_threshold=args.Vth, tau=args.tau)
    elif args.arch == 'vgg13':
        snn = spiking_vgg13_bn(neuron=neuron, num_classes=num_classes,
                               input_channels=input_channels, surrogate_function=sg, v_threshold=args.Vth, tau=args.tau)
    elif args.arch == 'vgg16':
        snn = spiking_vgg16_bn(neuron=neuron, num_classes=num_classes,
                               input_channels=input_channels, surrogate_function=sg, v_threshold=args.Vth, tau=args.tau)
    elif args.arch == 'vgg19':
        snn = spiking_vgg19_bn(neuron=neuron, num_classes=num_classes,
                               input_channels=input_channels, surrogate_function=sg, v_threshold=args.Vth, tau=args.tau)
    elif args.arch == 'dvsgestureNet':
        snn = dvsgestureNet(neuron=neuron, surrogate_function=sg, v_threshold=args.Vth, tau=args.tau)
    elif args.arch == 'resnet18_cifar':
        snn = resnet18_cifar(neuron=neuron, num_classes=num_classes,
                               input_channels=input_channels, surrogate_function=sg, v_threshold=args.Vth, tau=args.tau)
    elif args.arch == 'resnet19_cifar':
        snn = resnet19_cifar(neuron=neuron, num_classes=num_classes,
                               input_channels=input_channels, surrogate_function=sg, v_threshold=args.Vth, tau=args.tau)
    elif args.arch == 'resnet20_cifar_modified':
        snn = resnet20_cifar_modified(neuron=neuron, num_classes=num_classes,
                               input_channels=input_channels,surrogate_function=sg, v_threshold=args.Vth, tau=args.tau)
    else:
        raise NotImplementedError(f'暂未实现该网络结构{args.arch}')

    snn.to(device)

    # ======================================================================
    print(f'已选用数据集: {args.dataset}')
    print(f'已选用backbone: {args.arch}')
    print(f'已选用神经元类型: {args.neuron}')
    if args.neuron == 'LIF':
        print(f'LIF模型参数tau已设置为: {args.tau}')
    print(f"当前数据集: {args.dataset}, input_channels已设为 {input_channels},num_classes已设置为 {num_classes}")
    print(f"当前代理梯度已设置为: {args.surrogate}")
    if args.surrogate == 'adali':
        print(f'当前adaptive_function为 {args.adaptive}')
    # ======================================================================

    criterion_ce = nn.CrossEntropyLoss().to(device)
    criterion_mse = nn.MSELoss().to(device)
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(snn.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(snn.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)
    best_acc = 0
    best_epoch = 0
    for epoch in range(args.epochs):

        start_time = time.time()
        snn.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            functional.reset_net(snn)
            labels = labels.to(device)
            label_one_hot = F.one_hot(labels, num_classes).float()
            if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100' or args.dataset == 'TinyImageNet':
                images = images.repeat(args.step, 1, 1, 1, 1).to(device)
            elif args.dataset == 'CIFAR10DVS':
                images = torch.stack(images, dim=0).to(device)
            elif args.dataset == 'DVSGesture':
                images = images.transpose(0, 1).to(device)

            outputs = snn(images).mean(0)
            if args.loss == 'ce':
                loss = criterion_ce(outputs, labels)
            elif args.loss == 'mse':
                loss = criterion_mse(outputs,label_one_hot)
            elif args.loss == 'mix':
                loss = args.lambda*criterion_mse(outputs,label_one_hot)+(1-args.lambda)*criterion_ce(outputs, labels)

            if (i + 1) % 50 == 0:
                print("Loss: ", loss)
            loss.backward()
            optimizer.step()

        scheduler.step()
        if args.surrogate == 'adali':
            handleConfig(args.adaptive, epoch + 1)
        correct = 0
        total = 0
        acc = 0

        # start testing
        snn.eval()
        fr_monitor = monitor.OutputMonitor(snn, neuron, cal_firing_rate)
        mp_monitor = monitor.AttributeMonitor('v_seq', pre_forward=False, net=snn, instance=neuron)
        fr_monitor.disable()
        mp_monitor.disable()
        with (torch.no_grad()):
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                functional.reset_net(snn)

                if (epoch == 0 or (epoch + 1) % 100 == 0) and batch_idx == 0:
                    if args.show_fr:
                        fr_monitor.enable()
                        fr_monitor.records.clear()
                    if args.mp_statistic:
                        mp_monitor.enable()
                        mp_monitor.records.clear()
                        for m in snn.modules():
                            if isinstance(m, neuron):
                                m.store_v_seq = True

                if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100' or args.dataset == 'TinyImageNet':
                    inputs = inputs.repeat(args.step, 1, 1, 1, 1).to(device)
                elif args.dataset == 'CIFAR10DVS':
                    inputs = torch.stack(inputs, dim=0).to(device)
                elif args.dataset == 'DVSGesture':
                    inputs = inputs.transpose(0, 1).to(device)
                targets = targets.to(device)

                outputs = snn(inputs).mean(0)
                if (epoch == 0 or (epoch + 1) % 100 == 0) and batch_idx == 0:
                    if args.show_fr:
                        fr_monitor.disable()
                        print(f'firing rate: {sum(fr_monitor.records) / len(fr_monitor.records):.2f}')
                    if args.mp_statistic:
                        mp_monitor.disable()
                        for m in snn.modules():
                            if isinstance(m, neuron):
                                m.store_v_seq = False
                        min, q1, mean, q3, max, density = cal_mp_summary_statistics(mp_monitor.records)
                        print(
                            f'membrane potential: min = {min:.2f}, q1 = {q1:.2f}, mean = {mean:.2f}, q3 = {q3:.2f}, max = {max:.2f}')
                        print(f'compute density:{density:.2f}')

                _, predicted = outputs.cpu().max(1)
                total += (targets.size(0))
                correct += (predicted.eq(targets.cpu()).sum().item())

        acc = 100 * correct / total
        acc_np.append(round(acc, 2))
        np.save(acc_save_path, acc_np)
        print(f'Test Accuracy of the model on the test images: {acc}')
        if best_acc < acc:
            best_acc = acc
            best_epoch = epoch + 1
            torch.save(snn.state_dict(), model_save_name)
        print(f'best_acc is: {best_acc}')
        print(f'best_iter: {best_epoch}')
        print(f'Iters: {epoch + 1}\n')
