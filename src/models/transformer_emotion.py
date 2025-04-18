import torch
import torch.nn as nn

class TransformerEmotionModel(nn.Module):
    def __init__(self, input_dim=768, model_dim=512, num_heads=8, num_layers=4, dropout=0.1, task="expr"):
        super().__init__()

        self.task = task
        self.input_proj = nn.Linear(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(model_dim)

        if task == "expr":
            self.classifier = nn.Sequential(
                nn.LayerNorm(model_dim),
                nn.Linear(model_dim, 8)  # 8 emotion classes
            )
        elif task == "va":
            self.classifier = nn.Sequential(
                nn.LayerNorm(model_dim),
                nn.Linear(model_dim, 2),  # valence, arousal
                nn.Tanh()  # output range [-1, 1]
            )
        elif task == "au":
            self.classifier = nn.Sequential(
                nn.LayerNorm(model_dim),
                nn.Linear(model_dim, 12),  # 12 AU units
                nn.Sigmoid()  # binary activation per AU
            )
        else:
            raise ValueError(f"Unsupported task: {task}")

    def forward(self, x):
        # x: [B, T, 768]
        x = self.input_proj(x)          # [B, T, model_dim]
        x = self.encoder(x)             # [B, T, model_dim]
        x = x[:, -1]                    # use last time step: [B, model_dim]
        x = self.norm(x)
        out = self.classifier(x)
        return out

if __name__ == "__main__":
    task = "expr"  # expr,va or au
    model = TransformerEmotionModel(task=task)
    dummy = torch.randn(4, 30, 768)  # B=4, T=30, D=768
    out = model(dummy)
    print(f"Output shape for task '{task}':", out.shape)
    
    task = "va"
    model = TransformerEmotionModel(task=task)
    dummy = torch.randn(4, 30, 768)  # B=4, T=30, D=768
    out = model(dummy)
    print(f"Output shape for task '{task}':", out.shape)
    
    task = "au" 
    model = TransformerEmotionModel(task=task)
    dummy = torch.randn(4, 30, 768)  # B=4, T=30, D=768
    out = model(dummy)
    print(f"Output shape for task '{task}':", out.shape)