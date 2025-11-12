import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools


# Convolutional Block for shared Encoder
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        return self.conv(x)
    
    
# Dense Block for shared Encoder
class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        self.block1 = ConvBlock(in_channels, out_channels)
        self.block2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        # return downsampled feature map for skip connection to the Decoder and original map for Dense connection to next block
        return self.pool(x), x
    
    
# Soft Attention Module for the each Decoder Block
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # return feature map weighted by the attention map 
        return self.attention(x)
    
    
# Decoder Block with attention module
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(mid_channels + mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.attention = AttentionModule(out_channels)
        
    def forward(self, x, skip):
        x = self.upconv(x)  # Upsample features
        x = torch.cat([x, skip], dim=1)  # Concatenate the skip connection
        map1 = F.relu(self.bn1(self.conv1(x)))
        map2 = F.relu(self.bn2(self.conv2(map1)))
        attention_mask = self.attention(map1)
        
        return  map2 * attention_mask 
    
    
# AdaMT-Net Model combining the Encoder and Decoder Blocks in a U-Net architecture
class AdaMTNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=19):
        super(AdaMTNet, self).__init__()
        
        # Shared Encoder
        self.encoder_block1 = DenseBlock(in_channels, 64)
        self.encoder_block2 = DenseBlock(64, 128)
        self.encoder_block3 = DenseBlock(128, 256)
        self.encoder_block4 = DenseBlock(256, 512)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)
        
        # Decoder for Semantic Segmentation
        self.seg_decoder_block1 = DecoderBlock(1024, 512, 512)
        self.seg_decoder_block2 = DecoderBlock(512, 256, 256)
        self.seg_decoder_block3 = DecoderBlock(256, 128, 128)
        self.seg_decoder_block4 = DecoderBlock(128, 64, 64)
        self.seg_final = nn.Conv2d(64, num_classes, kernel_size=1)
        self.seg_upconv = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2)
        
        # Decoder for Depth Estimation
        self.depth_decoder_block1 = DecoderBlock(1024, 512, 512)
        self.depth_decoder_block2 = DecoderBlock(512, 256, 256)
        self.depth_decoder_block3 = DecoderBlock(256, 128, 128)
        self.depth_decoder_block4 = DecoderBlock(128, 64, 64)
        self.depth_final = nn.Conv2d(64, 1, kernel_size=1)
        self.depth_upconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
        
    @property
    def seg_decoder_parameters(self):
        return itertools.chain(
            self.seg_decoder_block1.parameters(),
            self.seg_decoder_block2.parameters(),
            self.seg_decoder_block3.parameters(),
            self.seg_decoder_block4.parameters(),
            self.seg_final.parameters()
        )
        
    @property
    def depth_decoder_parameters(self):
        return itertools.chain(
            self.depth_decoder_block1.parameters(),
            self.depth_decoder_block2.parameters(),
            self.depth_decoder_block3.parameters(),
            self.depth_decoder_block4.parameters(),
            self.depth_final.parameters()
        )
        
    def forward(self, x):
        # Shared Encoder
        x, skip1 = self.encoder_block1(x)
        x, skip2 = self.encoder_block2(x)
        x, skip3 = self.encoder_block3(x)
        x, skip4 = self.encoder_block4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder for Semantic Segmentation
        seg_x = self.seg_decoder_block1(x, skip4)
        seg_x = self.seg_decoder_block2(seg_x, skip3)
        seg_x = self.seg_decoder_block3(seg_x, skip2)
        seg_x = self.seg_decoder_block4(seg_x, skip1)
        seg_map = self.seg_final(seg_x)
        # seg_map = self.seg_upconv(seg_map)
        
        # Decoder for Depth Estimation
        depth_x = self.depth_decoder_block1(x, skip4)
        depth_x = self.depth_decoder_block2(depth_x, skip3)
        depth_x = self.depth_decoder_block3(depth_x, skip2)
        depth_x = self.depth_decoder_block4(depth_x, skip1)
        depth_map = self.depth_final(depth_x)
        # depth_map = self.depth_upconv(depth_map)
        
        return seg_map, depth_map
