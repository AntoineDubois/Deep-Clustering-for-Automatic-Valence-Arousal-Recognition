import torch


class EncoderGRU(torch.nn.Module):
    def __init__(self, seq_len: int, n_features: int, embedding_dim: int=2, hidden_dim: int=128):
        super(EncoderGRU, self).__init__()

        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        self.rnn1 = torch.nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
            )

        self.rnn2 = torch.nn.GRU(
            input_size=hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
            )

        self.norm1 = torch.nn.LayerNorm(hidden_dim)
        self.norm2 = torch.nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x, _ = self.rnn1(x)
        x = self.norm1(x) #
        x, hidden_n = self.rnn2(x)
        hidden_n = torch.transpose(hidden_n,1, 0)
        hidden_n = self.norm2(hidden_n)
        return hidden_n



class DecoderGRU(torch.nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1, hidden_dim=126):
        super(DecoderGRU, self).__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = input_dim
        self.n_features = n_features

        self.rnn1 = torch.nn.GRU(
            input_size=input_dim * n_features,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False
            )

        self.rnn2 = torch.nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False
            )

        self.norm1 = torch.nn.LayerNorm(self.hidden_dim)
        self.norm2 = torch.nn.LayerNorm(self.hidden_dim)

        self.output_layer = torch.nn.Linear(self.hidden_dim, n_features)
        
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = x.repeat(1,self.seq_len, self.n_features)
        x, _ = self.rnn1(x)
        x = self.norm1(x) #
        x, _ = self.rnn2(x)
        x = self.norm2(x)
        x = self.output_layer(x)

        return x


class StoSAutoencoder(torch.nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, hidden_dim=128):
        super(StoSAutoencoder, self).__init__()

        self.encoder = EncoderGRU(seq_len=seq_len, n_features=n_features, embedding_dim=embedding_dim, hidden_dim=128)
        self.decoder = DecoderGRU(seq_len=seq_len, input_dim=embedding_dim, n_features=n_features, hidden_dim=128)

    def forward(self, x):
        latent = self.encoder(x)
        x = self.decoder(latent)

        return x, latent