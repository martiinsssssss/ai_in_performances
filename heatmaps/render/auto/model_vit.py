# model_vit.py
import torch
import torch.nn as nn
import timm

class ViT(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=pretrained)
        self.input_size = (224, 224)
        encoder_output_channels = self.encoder.embed_dim

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(encoder_output_channels, 512, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.encoder.patch_embed(x)
        cls_token = self.encoder.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.encoder.pos_embed
        x = self.encoder.pos_drop(x)

        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        x = x[:, 1:, :]  # remove cls token

        patch_h, patch_w = self.encoder.patch_embed.patch_size
        H, W = self.input_size[0] // patch_h, self.input_size[1] // patch_w
        x = x.view(B, H, W, -1).permute(0, 3, 1, 2)

        return self.decoder(x)