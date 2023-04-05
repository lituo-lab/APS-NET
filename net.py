from torch import nn
import torch


class Feature_Encoder(nn.Module):
    def __init__(self, in_channel=1, out_channel=8):
        super(Feature_Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel*2, 5, 1, 2, padding_mode='reflect'),
            nn.BatchNorm2d(in_channel*2, track_running_stats=False),
            nn.PReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel*2, in_channel*4, 5, 1, 2, padding_mode='reflect'),
            nn.BatchNorm2d(in_channel*4, track_running_stats=False),
            nn.PReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel*4, out_channel, 5, 1, 2, padding_mode='reflect'),
            nn.BatchNorm2d(out_channel, track_running_stats=False),
            nn.PReLU())

        self.conv4 = nn.Conv2d(out_channel, out_channel, 5, 2, 2, padding_mode='reflect')

    def forward(self, x):
        y = self.conv3(self.conv2(self.conv1(x)))
        return self.conv4(y)


class Channel_Attention(nn.Module):

    def __init__(self, channel=8):
        super(Channel_Attention, self).__init__()
        self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.__max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.__sigmoid = nn.Sigmoid()

        self.__fc = nn.Sequential(
            nn.Conv2d(channel, 4, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(4, channel, kernel_size=1, bias=False))

    def forward(self, x):
        y1 = self.__avg_pool(x)
        y1 = self.__fc(y1)
        y2 = self.__max_pool(x)
        y2 = self.__fc(y2)
        y = self.__sigmoid(y1+y2)
        return x * y


class Spartial_Attention(nn.Module):

    def __init__(self):
        super(Spartial_Attention, self).__init__()
        self.__layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=5, padding=2, padding_mode='reflect'),
            nn.Sigmoid())

    def forward(self, x):
        avg_mask = torch.mean(x, dim=1, keepdim=True)
        max_mask, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_mask, max_mask], dim=1)
        mask = self.__layer(mask)
        return x * mask


class Down_Module(nn.Module):
    def __init__(self, in_channel=9, out_channel=8):
        super(Down_Module, self).__init__()

        self.pool = nn.MaxPool2d(2)

        self.line1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel//2, 5,  1, 2, padding_mode='reflect'),
            nn.Conv2d(out_channel//2, out_channel//2, 3, 1, 2, dilation=2, padding_mode='reflect'))

        self.line2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel//2, 3, 1, 2, dilation=2, padding_mode='reflect'),
            nn.Conv2d(out_channel//2, out_channel//2, 5, 1, 2, padding_mode='reflect'))

        self.conv = nn.Conv2d(in_channel+out_channel, out_channel, 5, 2, 2, padding_mode='reflect')

    def forward(self, img, x):
        y = torch.cat([img, x], dim=1)
        z = torch.cat([self.line1(y), self.line2(y), y], dim=1)
        return self.conv(z)


class Feature_Fusion_Module(nn.Module):

    def __init__(self, in_channel=17, out_channel=2):
        super(Feature_Fusion_Module, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 5, 1, 2, padding_mode='reflect'),
            nn.BatchNorm2d(in_channel, track_running_stats=False),
            nn.PReLU(),
            nn.Conv2d(in_channel, in_channel, 5, 1, 2, padding_mode='reflect'))

        self.line = nn.Sequential(
            nn.Conv2d(in_channel, 1, 1, bias=False),
            nn.Sigmoid())

        self.conv2 = nn.Conv2d(in_channel, out_channel, 5, 1, 2, padding_mode='reflect')
        self.Upsample = nn.Upsample(scale_factor=8, mode='bilinear')

    def forward(self, x_refine, x_spatial, x_short):

        x_refine = nn.MaxPool2d(4)(x_refine)
        y = self.conv(torch.cat([x_refine, x_spatial, x_short], dim=1))
        w = self.line(y)
        y = self.conv2(y + y*w)

        return self.Upsample(y)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.Feature_Encoder = Feature_Encoder()
        self.Channel_Attention = Channel_Attention()
        self.Spartial_Attention = Spartial_Attention()
        self.Down_Module = Down_Module()
        self.Feature_Fusion_Module = Feature_Fusion_Module()

    def forward(self, x):

        x_Feature = self.Feature_Encoder(x)

        x_short = nn.MaxPool2d(8)(x)
        x_refine = self.Spartial_Attention(self.Channel_Attention(x_Feature))

        x_spatial = self.Down_Module(nn.MaxPool2d(2)(x), x_Feature)
        x_spatial = self.Down_Module(nn.MaxPool2d(4)(x), x_spatial)

        out = self.Feature_Fusion_Module(x_refine, x_spatial, x_short)

        return out


if __name__ == '__main__':

    net1 = Net().to('cuda')
    x = torch.randn(2, 1, 1024, 1024).to('cuda')
    y = net1(x)
    print(x.shape)
    print(y.shape)
    # from torchstat import stat
    # stat(net1,(1, 1024, 1024))
    print(y[1, 1, 1, 1].item())
