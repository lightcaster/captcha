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
import models

from sklearn.metrics import confusion_matrix
from utils import plot_confusion_matrix

import matplotlib.pyplot as plt

def predict(model, device, dataset):
    model.eval()
    total_loss = 0
    char_acc_rate = 0
    sample_acc_rate = 0

    with torch.no_grad():
        for step, (images, labels) in enumerate(dataset):

            inputs, targets = images.to(device), labels.to(device)

            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
           
            total_loss += loss.data.cpu()

            dist = F.log_softmax(logits, dim=1)
            pred = dist.max(1)[1]

            hits = pred.eq(targets) 
            char_acc_rate += hits.sum()
            sample_acc_rate += hits.prod(1).sum()

    return total_loss, char_acc_rate

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, required=True,
            help='pretrained model')
    parser.add_argument('--data', '-d', type=str, required=True,
            help='data root directory')
    parser.add_argument('--batch-size', '-b', type=int, default=16,
                        help='input batch size')
    parser.add_argument('--confusion-matrix', '-cf', action='store_true', 
            default=False, help='disables CUDA training')
    parser.add_argument('--no-cuda', action='store_true', 
            default=False, help='disables CUDA training')

    args = parser.parse_args()

    device = torch.device("cpu" if args.no_cuda else "cuda")

    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = CaptchaDataset(
            root_dir=args.data,
            transform=data_transform)

    model = models.DeepCNN(
            length=dataset.label_length,
            n_classes = len(dataset.alphabet))

    model.load_state_dict(torch.load(args.model))
    model.to(device)

    data_loader = DataLoader(dataset, 
            batch_size=1,
            shuffle=False)

    result = []
    total_pred, total_given = [], []
    for i, (images, labels) in enumerate(data_loader):

        t0 = time.time()

        with torch.no_grad():
            inputs, targets = images.to(device), labels.to(device)
            logits = model(inputs)

        loss = F.cross_entropy(logits, targets)
        dist = F.log_softmax(logits, dim=1)
        pred = dist.max(1)[1]

        pred_label  = "".join(dataset.decode_label(pred[0]))

        total_pred.extend(pred[0].cpu().numpy())
        total_given.extend(labels[0].cpu().numpy())

        result.append((loss.data.cpu(), dataset.image_names[i], pred_label))

    for loss, name, label in sorted(result, key=lambda x: -x[0]):
        err = ' ' if label == name[:-4] else '*'
        print ("{}\t{}\t{}\t{:.8f}".format(name, label, err, loss))

    pr, gv = np.array(total_pred), np.array(total_given)
    car = np.mean(pr == gv)
    sar = np.mean(np.prod(pr.reshape(-1, 5), axis=1) == \
            np.prod(gv.reshape(-1,5), axis=1))

    print ("character accuracy rate:\t{:.4f}\n"
            "sample accuracy rate:\t\t{:.4f}".format(car, sar))

    if args.confusion_matrix:
        cnf_matrix = confusion_matrix(total_given, total_pred)
        plot_confusion_matrix(cnf_matrix, dataset.alphabet)
        plt.show()

