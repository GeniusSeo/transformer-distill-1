import torch
import torch.nn as nn
import random


class EncoderRNN(nn.Module):
    def __init__(self, embedding_size, input_size, hidden_size, layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=layers, dropout=dropout)

    def forward(self, input_word):
        embedding = self.embedding(input_word)
        output, (hidden, cell) = self.lstm(embedding)
        return hidden, cell


class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, input_size, hidden_size, output_size, layers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.layers = layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=layers, dropout=dropout)
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_word, hidden, cell):
        embedding = self.dropout(self.embedding(input_word))
        output, (hidden, cell) = self.lstm(embedding, (hidden, cell))
        output = self.output(output)
        output = self.softmax(output)
        return output, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.layers == decoder.layers, "encoder and decoder must have equal number of layers"

    def forward(self, input, target, teacher_forcing_ratio=0.5):
        batch_size = target.shape[1]
        seq_len = target.shape[0]
        output_size = self.decoder.output_size

        outputs = torch.zeros(seq_len, batch_size, output_size, device=self.device)
        hidden, cell = self.encoder(input)
        dec_input = target[0, :].unsqueeze(0)
        for i in range(seq_len):
            dec_output, hidden, cell = self.decoder(dec_input, hidden, cell)
            outputs[i] = dec_output
            if random.random() < teacher_forcing_ratio:
                dec_input = target[i].unsqueeze(0)
            else:
                dec_input = dec_output.argmax(2)
        return outputs


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
