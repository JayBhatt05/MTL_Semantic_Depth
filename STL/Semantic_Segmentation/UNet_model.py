# U-Net model for Semantic Segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, num_classes=19, in_channels=3, init_features=64):
        super().__init__()
        
        # Encoder path
        self.encoder1 = DoubleConv(in_channels, init_features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = DoubleConv(init_features, init_features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = DoubleConv(init_features * 2, init_features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = DoubleConv(init_features * 4, init_features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(init_features * 8, init_features * 16)
        
        # Decoder path
        self.upconv4 = nn.ConvTranspose2d(init_features * 16, init_features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(init_features * 16, init_features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(init_features * 8, init_features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(init_features * 8, init_features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(init_features * 4, init_features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(init_features * 2, init_features)
        
        # Final convolution to get to one channel (segmentation map)
        self.final_conv = nn.Conv2d(init_features, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # Final convolution and softmax activation
        seg_map = self.final_conv(dec1)
        
        return seg_map