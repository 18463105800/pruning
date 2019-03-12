"""
trainer for additionalNet
"""

import math
import time

import torch.autograd
import torch.nn as nn
from additional_classifier import AdditionalClassfier

from ori_model  import vgg

class View(nn.Module):
    """
    reshape data from 4 dimension to 2
    """

    def forward(self, x):
        if x.dim() == 4:
            out = x.view(x.size(0),-1)
        else:
            out = x
        return out

class DiscriminativeTrainer(object):
    """
    discriminative trainer for net
    """

    def __init__(self,ori_model,pruned_model,train_loader,val_loader,settings):

        self.ori_model = ori_model
        self.pruned_model = pruned_model
        self.settings = settings
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.ori_segments = []
        self.pruned_segments = []
        self.seg_optimizer = []
        self.aux_fc = []
        self.fc_optimizer = []

        self.preprocess()

    def preprocess(self):
        """
        split the network into serveral segments
        create auxiliary classifiers
        create optimizers for network segments and fcs
        """

        net_origin = None
        net_pruned = None


