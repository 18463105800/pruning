from ori_model.vgg import VGG
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from additional_classifier import AdditionalClassfier

x = Variable(torch.FloatTensor(16, 3, 40, 40))
model = VGG(isTraining = [1, 1, 1, 1, 0])


def train(self, epoch, index):
    """
    train
    :param epoch: index of epoch
    :param index: index of segment
    """

    iters = len(self.train_loader)

    num_segments = len(model.features)
    for i in range(num_segments):
        self.pruned_segments[i].train()
        if i != index and i != num_segments - 1:
            model.aux[i].eval()
        else:
            model.aux[i].train()

    for i, (images, labels) in enumerate(self.train_loader):

        if self.settings.n_gpus == 1:
            images = images.cuda()
        labels = labels.cuda()

        # forward
        outputs, losses = self.forward(images, labels)
        # backward
        self.backward(losses, index)