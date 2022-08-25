import torch
from torch.nn import functional


class InceptionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv11 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=(1, 1), stride=(stride, stride), padding=(0, 0), bias=False)

        self.conv21 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=(1, 1), stride=(stride, stride), padding=(0, 0), bias=False)
        self.conv22 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                      kernel_size=(5, 5), stride=(stride, stride), padding=(2, 2), bias=False)

        self.conv31 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=(1, 1), stride=(stride, stride), padding=(0, 0), bias=False)
        self.conv32 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                      kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False)
        self.conv33 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                      kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False)

        self.avg_pool41 = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv42 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                      kernel_size=(1, 1), stride=(stride, stride), padding=(0, 0), bias=False)

        self.bn = torch.nn.BatchNorm2d(num_features=out_channels * 4)

    def forward(self, x):
        out1 = self.conv11.forward(x)

        out2 = self.conv21.forward(x)
        out2 = self.conv22.forward(out2)

        out3 = self.conv31.forward(x)
        out3 = self.conv32.forward(out3)
        out3 = self.conv33.forward(out3)

        out4 = self.avg_pool41(x)
        out4 = self.conv42(out4)

        out = torch.cat([out1, out2, out3, out4], dim=1)
        # out = F.relu(out)
        out = functional.elu(out, alpha=1.1, inplace=True)
        out = self.bn.forward(out)

        return out


class InceptionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(1, 1), bias=False)
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.inception_block1 = InceptionBlock(in_channels=64, out_channels=128)

        self.conv2 = torch.nn.Conv2d(in_channels=128 * 4, out_channels=256, kernel_size=(3, 3), stride=(1, 1),
                                     bias=False)
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.inception_block2 = InceptionBlock(in_channels=256, out_channels=384)

        self.linear = torch.nn.Linear(384 * 4 * 133 * 133, 4)

    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.max_pool1.forward(out)
        # out = F.relu(out)
        out = functional.elu(out, alpha=1.1, inplace=True)

        out = self.inception_block1.forward(out)

        out = self.conv2.forward(out)
        out = self.max_pool2.forward(out)
        # out = F.relu(out)
        out = functional.elu(out, alpha=1.1, inplace=True)

        out = self.inception_block2.forward(out)

        out = out.view(-1, 384 * 4 * 133 * 133)
        out = self.linear.forward(out)
        out = functional.softmax(out, dim=1)

        return out
