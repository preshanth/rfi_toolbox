# rfi_toolbox/models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()
        self.conv = DoubleConv(in_channels, features)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x)), self.conv(x) # Returning skip connection as well

class Decoder(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, features, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, features)

    def forward(self, x, skip):
        up = self.up(x)
        concat = torch.cat([up, skip], dim=1)
        return self.conv(concat)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super().__init__()

        features = init_features
        self.encoder1 = Encoder(in_channels, features)
        self.encoder2 = Encoder(features, features * 2)
        self.encoder3 = Encoder(features * 2, features * 4)
        self.encoder4 = Encoder(features * 4, features * 8)

        self.bottleneck = DoubleConv(features * 8, features * 16)

        self.decoder4 = Decoder(features * 16, features * 8)
        self.decoder3 = Decoder(features * 8, features * 4)
        self.decoder2 = Decoder(features * 4, features * 2)
        self.decoder1 = Decoder(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1_pool, enc1 = self.encoder1(x)
        enc2_pool, enc2 = self.encoder2(enc1_pool)
        enc3_pool, enc3 = self.encoder3(enc2_pool)
        enc4_pool, enc4 = self.encoder4(enc3_pool)

        # Bottleneck
        bottleneck = self.bottleneck(enc4_pool)

        # Decoder
        dec4 = self.decoder4(bottleneck, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        # Final convolution
        return self.final_conv(dec1)

class UNetBigger(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super().__init__()
        features = init_features
        self.encoder1 = Encoder(in_channels, features)
        self.encoder2 = Encoder(features, features * 2)
        self.encoder3 = Encoder(features * 2, features * 4)
        self.encoder4 = Encoder(features * 4, features * 8)
        self.encoder5 = Encoder(features * 8, features * 16) # Added layer

        self.bottleneck = DoubleConv(features * 16, features * 32) # Adjusted bottleneck

        self.decoder5 = Decoder(features * 32, features * 16) # Added layer
        self.decoder4 = Decoder(features * 16, features * 8)
        self.decoder3 = Decoder(features * 8, features * 4)
        self.decoder2 = Decoder(features * 4, features * 2)
        self.decoder1 = Decoder(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1_pool, enc1 = self.encoder1(x)
        enc2_pool, enc2 = self.encoder2(enc1_pool)
        enc3_pool, enc3 = self.encoder3(enc2_pool)
        enc4_pool, enc4 = self.encoder4(enc3_pool)
        enc5_pool, enc5 = self.encoder5(enc4_pool) # Added layer

        # Bottleneck
        bottleneck = self.bottleneck(enc5_pool) # Adjusted bottleneck

        # Decoder
        dec5 = self.decoder5(bottleneck, enc5) # Added layer
        dec4 = self.decoder4(dec5, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        # Final convolution
        return self.final_conv(dec1)

class DoubleConvOverfit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), # Keep batch norm (can help with training stability)
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class EncoderOverfit(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()
        self.conv = DoubleConvOverfit(in_channels, features)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x)), self.conv(x)

class DecoderOverfit(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, features, kernel_size=2, stride=2)
        self.conv = DoubleConvOverfit(in_channels, features)

    def forward(self, x, skip):
        up = self.up(x)
        concat = torch.cat([up, skip], dim=1)
        return self.conv(concat)

class UNetOverfit(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=128): # Increased init_features significantly
        super().__init__()

        features = init_features
        self.encoder1 = EncoderOverfit(in_channels, features)
        self.encoder2 = EncoderOverfit(features, features * 2)
        self.encoder3 = EncoderOverfit(features * 2, features * 4)
        self.encoder4 = EncoderOverfit(features * 4, features * 8)
        self.encoder5 = EncoderOverfit(features * 8, features * 16) # Added a deeper layer

        self.bottleneck = DoubleConvOverfit(features * 16, features * 32) # Increased bottleneck size

        self.decoder5 = DecoderOverfit(features * 32, features * 16)
        self.decoder4 = DecoderOverfit(features * 16, features * 8)
        self.decoder3 = DecoderOverfit(features * 8, features * 4)
        self.decoder2 = DecoderOverfit(features * 4, features * 2)
        self.decoder1 = DecoderOverfit(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid() # Assuming binary segmentation

    def forward(self, x):
        # Encoder
        enc1_pool, enc1 = self.encoder1(x)
        enc2_pool, enc2 = self.encoder2(enc1_pool)
        enc3_pool, enc3 = self.encoder3(enc2_pool)
        enc4_pool, enc4 = self.encoder4(enc3_pool)
        enc5_pool, enc5 = self.encoder5(enc4_pool)

        # Bottleneck
        bottleneck = self.bottleneck(enc5_pool)

        # Decoder
        dec5 = self.decoder5(bottleneck, enc5)
        dec4 = self.decoder4(dec5, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        # Final convolution and activation
        return self.sigmoid(self.final_conv(dec1))

class DoubleConvDifferentActivation(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class EncoderDifferentActivation(nn.Module):
    def __init__(self, in_channels, features, activation=nn.ReLU):
        super().__init__()
        self.conv = DoubleConvDifferentActivation(in_channels, features, activation)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(x)), self.conv(x)

class DecoderDifferentActivation(nn.Module):
    def __init__(self, in_channels, features, activation=nn.ReLU):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, features, kernel_size=2, stride=2)
        self.conv = DoubleConvDifferentActivation(in_channels, features, activation)

    def forward(self, x, skip):
        up = self.up(x)
        concat = torch.cat([up, skip], dim=1)
        return self.conv(concat)

class UNetDifferentActivation(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32, activation=nn.ReLU):
        super().__init__()
        features = init_features
        self.encoder1 = EncoderDifferentActivation(in_channels, features, activation)
        self.encoder2 = EncoderDifferentActivation(features, features * 2, activation)
        self.encoder3 = EncoderDifferentActivation(features * 2, features * 4, activation)
        self.encoder4 = EncoderDifferentActivation(features * 4, features * 8, activation)

        self.bottleneck = DoubleConvDifferentActivation(features * 8, features * 16, activation)

        self.decoder4 = DecoderDifferentActivation(features * 16, features * 8, activation)
        self.decoder3 = DecoderDifferentActivation(features * 8, features * 4, activation)
        self.decoder2 = DecoderDifferentActivation(features * 4, features * 2, activation)
        self.decoder1 = DecoderDifferentActivation(features * 2, features, activation)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1_pool, enc1 = self.encoder1(x)
        enc2_pool, enc2 = self.encoder2(enc1_pool)
        enc3_pool, enc3 = self.encoder3(enc2_pool)
        enc4_pool, enc4 = self.encoder4(enc3_pool)

        # Bottleneck
        bottleneck = self.bottleneck(enc4_pool) 

        # Decoder
        dec4 = self.decoder4(bottleneck, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        # Final convolution
        return self.final_conv(dec1)

if __name__ == '__main__':
    # Example usage:
    x = torch.randn((1, 8, 512, 512)) # Batch size 1, 8 channels, 512x512
    #model = UNet(in_channels=8, out_channels=1, init_features=64)
    model = UNetBigger(in_channels=8, out_channels=1, init_features=64)
    model_overfit = UNetOverfit(in_channels=8, out_channels=1, init_features=128)    
    output = model_overfit(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
