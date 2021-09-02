import random
from typing import Tuple
from os.path import isfile
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, seq_dim: int, enc_hid_dim: int, dec_hid_dim: int):
        super().__init__()

        self.seq_dim = seq_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.rnn = nn.LSTM(seq_dim,
                           enc_hid_dim,
                           bidirectional=True,
                           num_layers=2)
        self.fc1 = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc2 = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor]:
        outputs, (hidden, cell) = self.rnn(src.float())
        hidden = torch.tanh(
            self.fc1(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        cell = torch.tanh(
            self.fc2(torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)))

        return outputs, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, seq_dim: int, enc_hid_dim: int, dec_hid_dim: int):
        super().__init__()

        self.seq_dim = seq_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.rnn = nn.LSTM(enc_hid_dim + seq_dim, dec_hid_dim, num_layers=2)
        self.out = nn.Linear(dec_hid_dim + seq_dim, seq_dim)

    def forward(self, input: torch.Tensor, decoder_hidden: torch.Tensor,
                decoder_cell: torch.Tensor,
                encoder_hidden: torch.Tensor) -> Tuple[torch.Tensor]:

        input = input.unsqueeze(0).float()
        encoder_hidden = encoder_hidden.unsqueeze(0).float()

        if type(decoder_hidden) != type(None):
            output, (decoder_hidden, decoder_cell) = self.rnn(
                torch.cat((input, encoder_hidden), dim=2),
                (decoder_hidden, decoder_cell))
        else:
            output, (decoder_hidden, decoder_cell) = self.rnn(
                torch.cat((input, encoder_hidden), dim=2))

        input = input.squeeze(0)
        output = output.squeeze(0)

        output = self.out(torch.cat((output, input), dim=1))

        return output, (decoder_hidden, decoder_cell)


class PairedSeqLSTM(nn.Module):
    def __init__(self,
                 seq_dim: int = 23,
                 enc_hid_dim: int = 64,
                 dec_hid_dim: int = 64):
        super().__init__()

        self.encoder = Encoder(seq_dim, enc_hid_dim, dec_hid_dim)
        self.decoder = Decoder(seq_dim, enc_hid_dim, dec_hid_dim)

    def forward(self,
                src: torch.Tensor,
                trg: torch.Tensor,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:

        device = src.device

        batch_size = src.shape[1]
        max_len = src.shape[0]
        seq_dim = src.shape[2]
        outputs = torch.zeros(max_len, batch_size, seq_dim).to(device)

        encoder_outputs, (encoder_hidden, _) = self.encoder(src)

        output = trg[0, :]
        hidden, cell = None, None
        for t in range(1, max_len):
            output, (hidden, cell) = self.decoder(output, hidden, cell,
                                                  encoder_hidden)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = F.one_hot(output.argmax(-1), num_classes=output.shape[1])
            output = (trg[t] if teacher_force else top1)

        return outputs


def load_model(model_file, eval_mode=True):
    if not isfile(model_file):
        raise FileNotFoundError("No file at {}".format(model_file))
    checkpoint_dict = torch.load(model_file, map_location='cpu')
    model_state = checkpoint_dict['model_state_dict']

    seq_dim = checkpoint_dict['seq_dim']
    enc_hid_dim = checkpoint_dict['enc_hid_dim']
    dec_hid_dim = checkpoint_dict['dec_hid_dim']

    model = PairedSeqLSTM(seq_dim=seq_dim,
                          enc_hid_dim=enc_hid_dim,
                          dec_hid_dim=dec_hid_dim)

    model.load_state_dict(model_state)

    if eval_mode:
        model.eval()

    return model
