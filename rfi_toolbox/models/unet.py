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

if __name__ == '__main__':
    # Example usage:
    x = torch.randn((1, 8, 512, 512)) # Batch size 1, 8 channels, 512x512
    model = UNet(in_channels=8, out_channels=1, init_features=32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
