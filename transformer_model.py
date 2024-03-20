import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from positional_encoding import PositionalEncoding  # Make sure to import from the correct file or directory
from torch.nn import TransformerDecoder, TransformerDecoderLayer



class TransformerModel(nn.Module):
    def __init__(self, seq_len, embedding_size, nhead, num_encoder_layers, num_classes, cfg):
        super(TransformerModel, self).__init__()
        self.cfg = cfg
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_classes = num_classes
        self.positional_encoding = PositionalEncoding(self.embedding_size)
        self.encoder_layer = TransformerEncoderLayer(d_model=self.embedding_size, nhead=self.nhead, batch_first=True, dropout=0.5)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=self.num_encoder_layers)
        if cfg['multi']:
            self.Linear = nn.Sequential(
            nn.Linear(self.embedding_size, self.num_classes))

        
        else:
            self.Linear = nn.Sequential(
            nn.Linear(self.seq_len * self.embedding_size, 1024),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, self.num_classes))

    def forward(self, x):
        # x: (batch_size, seq_len, embedding_size)
        self.batch_size = x.shape[0]
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        if not self.cfg['multi']:
            x = x.view(self.batch_size, -1)
        # else:
        #     x = torch.mean(x, dim=1)
        x = self.Linear(x)
        if self.cfg['task'] == "VA":
            x = torch.tanh(x)
        elif self.cfg['task'] == "EXPR":
            x = torch.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)
        
        x = x.view(-1, self.seq_len, self.num_classes)
        return x

class TransformerModel_decoder(nn.Module):
    def __init__(self, seq_len, nhead, embedding_size, num_encoder_layers, num_decoder_layers, cfg, feature_dim):
        super(TransformerModel_decoder, self).__init__()
        self.cfg = cfg
        self.feature_dim = feature_dim
        self.embedding_size = embedding_size
        self.seq_len = seq_len
        # Positional Encoding for time-series data
        self.positional_encoding = PositionalEncoding(self.embedding_size)
        # Transformer Encoder
        encoder_layers = TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        # Transformer Decoder
        decoder_layers = TransformerDecoderLayer(d_model=feature_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_layers=num_decoder_layers)
        # Output layer
        self.output_layer = nn.Linear(feature_dim, feature_dim)
    def forward(self, src, tgt):
        # Add positional encoding
        src = src + self.positional_encoding[:, :src.size(1)]
        tgt = tgt + self.positional_encoding[:, :tgt.size(1)]
        # Passing through the encoder
        memory = self.transformer_encoder(src)
        # Passing through the decoder
        output = self.transformer_decoder(tgt, memory)
        # Project back to feature space
        output = self.output_layer(output)
        
        return output
