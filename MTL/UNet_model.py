# U-Net model for MTL Setup with Semantic Segmentation and Depth Estimation tasks

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
    
# class SoftAttention(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.attn = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=1),
#             nn.BatchNorm2d(in_channels),
#             nn.Sigmoid()
#         )
        
#     def forward(self, x):
#         return x * self.attn(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))
    
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class UNet(nn.Module):
    def __init__(self, num_classes=19, in_channels=3, init_features=64):
        super().__init__()
        
        # Shared Encoder path
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
        
        # Attention
        # Attention Modules
        self.att4 = CBAM(init_features * 8)
        self.att3 = CBAM(init_features * 4)
        self.att2 = CBAM(init_features * 2)
        self.att1 = CBAM(init_features)
        
        # Task Specific Decoders path
        self.seg_upconv4 = nn.ConvTranspose2d(init_features * 16, init_features * 8, kernel_size=2, stride=2)
        self.seg_decoder4 = DoubleConv(init_features * 16, init_features * 8)
        self.depth_upconv4 = nn.ConvTranspose2d(init_features * 16, init_features * 8, kernel_size=2, stride=2)
        self.depth_decoder4 = DoubleConv(init_features * 16, init_features * 8)
        
        self.seg_upconv3 = nn.ConvTranspose2d(init_features * 8, init_features * 4, kernel_size=2, stride=2)
        self.seg_decoder3 = DoubleConv(init_features * 8, init_features * 4)
        self.depth_upconv3 = nn.ConvTranspose2d(init_features * 8, init_features * 4, kernel_size=2, stride=2)
        self.depth_decoder3 = DoubleConv(init_features * 8, init_features * 4)
        
        self.seg_upconv2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.seg_decoder2 = DoubleConv(init_features * 4, init_features * 2)
        self.depth_upconv2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.depth_decoder2 = DoubleConv(init_features * 4, init_features * 2)
        
        self.seg_upconv1 = nn.ConvTranspose2d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.seg_decoder1 = DoubleConv(init_features * 2, init_features)
        self.depth_upconv1 = nn.ConvTranspose2d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.depth_decoder1 = DoubleConv(init_features * 2, init_features)
        
        # Final convolution to get to one channel (segmentation map)
        self.seg_final_conv = nn.Conv2d(init_features, num_classes, kernel_size=1)
        self.depth_final_conv = nn.Conv2d(init_features, 1, kernel_size=1)

    def forward(self, x):
        # Shared Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Attention for Skip Connections
        enc4 = self.att4(enc4)
        enc3 = self.att3(enc3)
        enc2 = self.att2(enc2)
        enc1 = self.att1(enc1)
        
        # Task Specific Decoders with skip connections
        seg_dec4, depth_dec4 = self.seg_upconv4(bottleneck), self.depth_upconv4(bottleneck)
        seg_dec4, depth_dec4 = torch.cat((seg_dec4, enc4), dim=1), torch.cat((depth_dec4, enc4), dim=1)
        seg_dec4, depth_dec4 = self.seg_decoder4(seg_dec4), self.depth_decoder4(depth_dec4)
        
        seg_dec3, depth_dec3 = self.seg_upconv3(seg_dec4), self.depth_upconv3(depth_dec4)
        seg_dec3, depth_dec3 = torch.cat((seg_dec3, enc3), dim=1), torch.cat((depth_dec3, enc3), dim=1)
        seg_dec3, depth_dec3 = self.seg_decoder3(seg_dec3), self.depth_decoder3(depth_dec3)
        
        seg_dec2, depth_dec2 = self.seg_upconv2(seg_dec3), self.depth_upconv2(depth_dec3)
        seg_dec2, depth_dec2 = torch.cat((seg_dec2, enc2), dim=1), torch.cat((depth_dec2, enc2), dim=1)
        seg_dec2, depth_dec2 = self.seg_decoder2(seg_dec2), self.depth_decoder2(depth_dec2)
        
        seg_dec1, depth_dec1 = self.seg_upconv1(seg_dec2), self.depth_upconv1(depth_dec2)
        seg_dec1, depth_dec1 = torch.cat((seg_dec1, enc1), dim=1), torch.cat((depth_dec1, enc1), dim=1)
        seg_dec1, depth_dec1 = self.seg_decoder1(seg_dec1), self.depth_decoder1(depth_dec1)
        
        # Final convolution
        seg_map, depth_map = self.seg_final_conv(seg_dec1), self.depth_final_conv(depth_dec1)
        
        return seg_map, depth_map