import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CaptchaDataset
from models import SimpleCNN


def train(model, device, train_loader, optimizer):

        char_acc_rate = 0
        sample_acc_rate = 0
        total_loss = 0

        for step, (images, labels) in enumerate(train_loader):

            inputs, targets = images.to(device), labels.to(device)

            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data.cpu()

            dist = F.log_softmax(logits, dim=1)
            pred = dist.max(1)[1]

            hits = pred.eq(targets) 
            char_acc_rate += hits.sum()
            sample_acc_rate += hits.prod(1).sum()

        return total_loss, char_acc_rate, sample_acc_rate


def test(model, device, test_loader):
    model.eval()
    total_loss = 0
    char_acc_rate = 0

    with torch.no_grad():
        for step, (images, labels) in enumerate(test_loader):

            inputs, targets = images.to(device), labels.to(device)

            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
           
            total_loss += loss.data.cpu()

            dist = F.log_softmax(logits, dim=1)
            pred = dist.max(1)[1]

            hits = pred.eq(targets) 
            char_acc_rate += hits.sum()

    return total_loss, char_acc_rate

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default=None,
            help='pretrained model')
    parser.add_argument('--data', '-d', type=str, required=True,
            help='data root directory')
    parser.add_argument('--epochs', '-e', type=int, default=30, 
            help='number of epochs to train')
    parser.add_argument('--batch-size', '-b', type=int, default=16,
                        help='input batch size')
    parser.add_argument('--no-cuda', action='store_true', 
            default=False, help='disables CUDA training')

    args = parser.parse_args()

    device = torch.device("cpu" if args.no_cuda else "cuda")

    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CaptchaDataset(
            root_dir=os.path.join(args.data, 'train'), 
            transform=data_transform)

    test_dataset = CaptchaDataset(
            root_dir=os.path.join(args.data, 'test'), 
            transform=data_transform)

    model = SimpleCNN(
            length=train_dataset.label_length,
            n_classes = len(train_dataset.alphabet))

    if args.model:
            print('Loading model')
            model.load_state_dict(torch.load(args.model))

    model.to(device)

    train_loader = DataLoader(train_dataset, 
            batch_size=args.batch_size,
            shuffle=True)

    test_loader = DataLoader(test_dataset, 
            batch_size=args.batch_size,
            shuffle=False)

    optimizer = optim.Adadelta(model.parameters(), eps=1e-7)

    for epoch in range(args.epochs):

        t0 = time.time()
        loss, char_rate, sample_rate = train(model, device, train_loader, optimizer)

        print("train {0}: time: {1:.4f}, loss: {2:.4f}, char_acc_rate: {3:.4f} ".format(
                epoch, time.time() - t0, 
                loss / len(train_dataset),
                char_rate.float() / (len(train_dataset) * model.length)))

        if epoch and (epoch % 3) == 0:
            loss, char_rate = test(model, device, test_loader)

            print("!test {0}: time: {1:.4f}, loss: {2:.4f}, char_acc_rate: {3:.4f} ".format(
                epoch, time.time() - t0, 
                loss / len(test_dataset),
                char_rate.float() / (len(test_dataset) * model.length)))

    print('Saving model')
    torch.save(model.state_dict(), 'model.pth')

