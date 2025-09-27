import torch
import torch.nn as nn
import torch.nn.functional as F

class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self._initialize_weights()

    def forward(self, x):
        return self.down_conv(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self._initialize_weights()

    def forward(self, x):
        return self.up_conv(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class UpConv_last(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv_last, self).__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 32, kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self._initialize_weights()

    def forward(self, x):
        return self.up_conv(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.begin = nn.Conv2d(1, 16, kernel_size=3,padding=1)
        self.down1 = DownConv(16,32)
        self.down2 = DownConv(32, 64)
        self.down3 = DownConv(64, 128)
        self.down4 = DownConv(128, 256)
        self.down5 = DownConv(256, 256)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.up1 = UpConv(256,256)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.up2 = UpConv(512,128)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.up3 = UpConv(256,64)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.up4 = UpConv(128,32)
        self.up5 = UpConv_last(64,16)
        self.end = nn.Conv2d(16, 2, kernel_size=3,padding=1)

    def forward(self,x):
        x = self.begin(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(self.upsample1(x5))
        x = torch.cat([x, x4], dim=1)
        x = self.up2(self.upsample2(x))
        x = torch.cat([x,x3],dim=1)
        x = self.up3(self.upsample3(x))
        x = torch.cat([x, x2], dim=1)
        x = self.up4(self.upsample4(x))
        x = torch.cat([x, x1], dim=1)
        x = self.up5(x)
        x = self.end(x)
        return x

