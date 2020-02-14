import torch
import torch.nn as nn
import torch.nn.functional as F

from ResNet import resnet18

import pdb


# TODO
# add SE block
# pretrained weights
# think more about the self.denses


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.resnet1 = resnet18(input_channels=5)
        self.resnet2 = resnet18(input_channels=4)
        self.resnet3 = resnet18(input_channels=5)
        self.conv1 = nn.Conv2d(3 * 512, 512, kernel_size=1, stride=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, 300)

        self.denses = nn.Sequential(
            nn.Linear(512, 400),
            nn.Dropout(0.5),
            nn.Linear(400, 300),
        )

    def forward(self, x):
        # upper joint
        x1 = torch.cat((torch.unsqueeze(x[:, 0, :, :], 1),
                        torch.unsqueeze(x[:, 2, :, :], 1),
                        torch.unsqueeze(x[:, 5, :, :], 1),
                        torch.unsqueeze(x[:, 8, :, :], 1),
                        torch.unsqueeze(x[:, 11, :, :], 1)
                        ), 1)

        # middle joint
        x2 = torch.cat((torch.unsqueeze(x[:, 3, :, :], 1),
                        torch.unsqueeze(x[:, 6, :, :], 1),
                        torch.unsqueeze(x[:, 9, :, :], 1),
                        torch.unsqueeze(x[:, 12, :, :], 1)
                        ), 1)

        # lower joint
        x3 = torch.cat((torch.unsqueeze(x[:, 1, :, :], 1),
                        torch.unsqueeze(x[:, 4, :, :], 1),
                        torch.unsqueeze(x[:, 7, :, :], 1),
                        torch.unsqueeze(x[:, 10, :, :], 1),
                        torch.unsqueeze(x[:, 13, :, :], 1)
                        ), 1)

        fm1 = self.resnet1(x1)  # output is torch.Size([batch_size, 512, h, w])
        fm2 = self.resnet2(x2)
        fm3 = self.resnet3(x3)

        fm = torch.cat((fm1, fm2, fm3), 1)  # torch.Size([batch_size, 1536, h, w])
        out = self.conv1(fm)
        out = self.avgpool(out)  # torch.Size([batch_size, 512, 1, 1])
        out = torch.flatten(out, 1)  # torch.Size([batch_size, 512])

        # out = self.fc(out)                # use dense layer instead
        out = self.denses(out)

        return out
