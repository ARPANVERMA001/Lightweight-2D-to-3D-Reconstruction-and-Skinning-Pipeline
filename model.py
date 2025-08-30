import torch
import torch.nn as nn
import timm


class CNNTransformer3D(nn.Module):
    """
    Hybrid CNN + Transformer model for 2D-to-3D point cloud prediction.
    - CNN backbone extracts image features.
    - Transformer encoder builds global context over image patches.
    - Transformer decoder uses learnable 3D queries to predict XYZ coords.
    """
    def __init__(
        self,
        cnn_backbone_name: str = 'resnet50',
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_points: int = 1024,
    ):
        super().__init__()
        # 1) CNN backbone (features only)
        self.cnn = timm.create_model(cnn_backbone_name, pretrained=True, features_only=True)
        feat_channels = self.cnn.feature_info.channels()[-1]
        # project CNN features to d_model
        self.conv_proj = nn.Conv2d(feat_channels, d_model, kernel_size=1)

        # 2) Positional embeddings for flattened tokens
        # will be initialized in forward when spatial dims known
        self.pos_embed = None

        # 3) Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # 4) Learnable 3D point queries + decoder
        self.query_embed = nn.Parameter(torch.randn(num_points, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # 5) MLP head to predict XYZ from decoder outputs
        self.head = nn.Linear(d_model, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        B = x.size(0)
        # CNN feature map
        feats = self.cnn(x)[-1]  # (B, C, H', W')
        feats = self.conv_proj(feats)  # (B, D, H', W')

        # flatten spatial dims
        D, Hp, Wp = feats.shape[1:]
        N = Hp * Wp
        tokens = feats.flatten(2).transpose(1, 2)  # (B, N, D)

        # initialize positional embeddings if needed
        if self.pos_embed is None or self.pos_embed.shape[0] != N:
            self.pos_embed = nn.Parameter(torch.randn(N, D, device=x.device))
        tokens = tokens + self.pos_embed.unsqueeze(0)

        # transformer encoder expects (seq_len, batch, dim)
        enc = self.transformer_encoder(tokens.transpose(0, 1))  # (N, B, D)

        # prepare queries: (num_points, B, D)
        q = self.query_embed.unsqueeze(1).repeat(1, B, 1)

        # transformer decoder with cross-attention
        dec = self.transformer_decoder(tgt=q, memory=enc)  # (num_points, B, D)

        # predict XYZ
        coords = self.head(dec.transpose(0, 1))  # (B, num_points, 3)
        return coords


if __name__ == '__main__':
    # quick sanity test
    model = CNNTransformer3D()
    img = torch.randn(2, 3, 224, 224)
    pts = model(img)
    print(pts.shape)  # should be [2, 1024, 3]
