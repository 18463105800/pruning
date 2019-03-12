# encoding:utf-8
import torch.nn as nn

class AdditionalClassfier(nn.Module):
    """
    定义额外的辅助层
    辅助层结构BN->Relu->AvgPooling->softmax(Fc)
    """

    def __init__(self,in_channels,num_classes):
        super(AdditionalClassfier,self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_channels,num_classes)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()
        self.fc.bias.data.zero_()

    def forward(self, x):

        x = self.bn(x)
        x = self.relu(x)
        # net的结构一般是(N,n,h,w)，mean(2)就变成(N,n,w),再次mean(2)就变成(N,n)
        x = x.mean(2).mean(2)
        output = self.fc(x)

        return output