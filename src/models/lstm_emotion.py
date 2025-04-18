import torch
import torch.nn as nn

class LSTMEmotionModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2, dropout=0.1, task="expr"):
        super().__init__()

        self.task = task
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )

        self.norm = nn.LayerNorm(hidden_dim)

        if task == "expr":
            self.classifier = nn.Linear(hidden_dim, 8)
        elif task == "va":
            self.classifier = nn.Linear(hidden_dim, 2)
        elif task == "au":
            self.classifier = nn.Linear(hidden_dim, 12)
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, x):
        # x: [B, T, 768]
        x, _ = self.lstm(x)             # [B, T, hidden_dim]
        x = x[:, -1]                    # last time step
        x = self.norm(x)
        out = self.classifier(x)
        return out

if __name__ == "__main__":
    task="expr"
    model = LSTMEmotionModel(task=task)
    dummy = torch.randn(4, 30, 768)  # B=4, T=30, D=768
    out = model(dummy)
    print(f"Output shape for task '{task}':", out.shape)

    task = "va"
    model = LSTMEmotionModel(task=task)
    dummy = torch.randn(4, 30, 768)  # B=4, T=30, D=768
    out = model(dummy)
    print(f"Output shape for task '{task}':", out.shape)
    
    task = "au" 
    model = LSTMEmotionModel(task=task)
    dummy = torch.randn(4, 30, 768)  # B=4, T=30, D=768
    out = model(dummy)
    print(f"Output shape for task '{task}':", out.shape)