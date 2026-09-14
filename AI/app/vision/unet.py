import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet(nn.Module):
    """Lightweight U-Net model for real-time Face Segmentation."""
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super(UNet, self).__init__()
        
        # Encoder (Downsampling)
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(32, 64)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(64, 128)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(128, 256)
        )
        
        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)
        
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder with skip connections
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        
        logits = self.outc(x)
        return self.sigmoid(logits)
