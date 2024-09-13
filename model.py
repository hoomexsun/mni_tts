from math import sqrt

from torch import nn, Tensor
from torch.nn import functional as F
import torch

from layers import ConvNorm, LinearNorm
from utils import get_mask_from_lengths, to_gpu


class LocationLayer(nn.Module):

    def __init__(
        self, attention_n_filters: int, attention_kernel_size: int, attention_dim: int
    ) -> None:
        super().__init__()
        padding = (attention_kernel_size - 1) // 2
        self.location_conv = ConvNorm(
            in_channels=2,
            out_channels=attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1,
        )
        self.location_dense = LinearNorm(
            in_dim=attention_n_filters,
            out_dim=attention_dim,
            bias=False,
            w_init_gain="tanh",
        )

    def forward(self, attention_weights_cat: Tensor) -> Tensor:
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        return self.location_dense(processed_attention)


class Attention(nn.Module):

    def __init__(
        self,
        attention_rnn_dim: int,
        embedding_dim: int,
        attention_dim: int,
        attention_location_n_filters: int,
        attention_location_kernel_size: int,
    ) -> None:
        super().__init__()
        self.query_layer = LinearNorm(
            in_dim=attention_rnn_dim,
            out_dim=attention_dim,
            bias=False,
            w_init_gain="tanh",
        )
        self.memory_layer = LinearNorm(
            in_dim=embedding_dim,
            out_dim=attention_dim,
            bias=False,
            w_init_gain="tanh",
        )
        self.v = LinearNorm(in_dim=attention_dim, out_dim=1, bias=False)
        self.location_layer = LocationLayer(
            attention_n_filters=attention_location_n_filters,
            attention_kernel_size=attention_location_kernel_size,
            attention_dim=attention_dim,
        )
        self.score_mark_value = -float("inf")

    def get_alignment_energies(
        self,
        query: Tensor,
        processed_memory: Tensor,
        attention_weights_cat: Tensor,
    ):
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(
            torch.tanh(processed_query + processed_attention_weights + processed_memory)
        )
        return energies.squeeze(-1)


class Prenet(nn.Module):

    def __init__(self, in_dim, sizes) -> None:
        super().__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [
                LinearNorm(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):

    def __init__(self, hparams) -> None:
        super().__init__()
        self.convolutions = nn.ModuleList()

        # First convolution layer
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    in_channels=hparams.n_mel_channels,
                    out_channels=hparams.postnet_embedding_dim,
                    kernel_size=hparams.postnet_kernel_size,
                    stride=1,
                    padding=(hparams.postnet_kernel_size - 1) // 2,
                    dilation=1,
                    w_init_gain="tanh",
                ),
                nn.BatchNorm1d(hparams.postnet_embedding_dim),
            )
        )

        # Middle convolution layers
        for _ in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        in_channels=hparams.postnet_embedding_dim,
                        out_channels=hparams.postnet_embedding_dim,
                        kernel_size=hparams.postnet_kernel_size,
                        stride=1,
                        padding=(hparams.postnet_kernel_size - 1) // 2,
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim),
                )
            )

        # Last convolution layer
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    in_channels=hparams.postnet_embedding_dim,
                    out_channels=hparams.n_mel_channels,
                    kernel_size=hparams.postnet_kernel_size,
                    stride=1,
                    padding=(hparams.postnet_kernel_size - 1) // 2,
                    dilation=1,
                    w_init_gain="linear",
                ),
                nn.BatchNorm1d(hparams.n_mel_channels),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        for conv_layer in self.convolutions[:-1]:
            x = F.dropout(F.tanh(conv_layer(x)), p=0.5, training=self.training)
        x = F.dropout(self.convolutions[-1](x), p=0.5, training=self.training)
        return x


class Encoder(nn.Module):
    """Encoder module:

    3 x (ConvNorm + BatchNorm1d + <relu + dropout0.5>) + <flatten> + BiLSTM

    Hyper-parameters:
        hparams.encoder_n_convolutions <- 3
        hparams.encoder_embedding_dim <- 512
        hparams.encoder_kernel_size <- 5
    """

    def __init__(self, hparams) -> None:
        super().__init__()
        self.convolutions = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNorm(
                        in_channels=hparams.encoder_embedding_dim,
                        out_channels=hparams.encoder_embedding_dim,
                        kernel_size=hparams.encoder_kernel_size,
                        stride=1,
                        padding=int((hparams.encoder_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="relu",
                    ),
                    nn.BatchNorm1d(hparams.encoder_embeddng_dim),
                )
                for _ in range(hparams.encoder_n_convolutions)
            ]
        )

        self.lstm = nn.LSTM(
            input_size=hparams.encoder_embedding_dim,
            hidden_size=int(hparams.encoder_embedding_dim / 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    # 3 * (x -> conv -> relu -> dropout (0.5)) ->
    # pack_padded -> flatten -> bilstm -> pad_packed
    def forward(self, x: Tensor, input_lengths: Tensor) -> Tensor:
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs

    def inference(self, x: Tensor) -> Tensor:
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return outputs


class Decoder(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim],
        )

        self.attention_rnn = nn.LSTMCell(
            input_size=hparams.prenet_dim + hparams.encoder_embedding_dim,
            hidden_size=hparams.attention_rnn_dim,
        )

        self.attention_layer = Attention(
            attention_rnn_dim=hparams.attention_rnn_dim,
            embedding_dim=hparams.encoder_embedding_dim,
            attention_dim=hparams.attention_dim,
            attention_location_n_filters=hparams.attention_location_n_filters,
            attention_location_kernel_size=hparams.attention_location_kernel_size,
        )

        self.decoder_rnn = nn.LSTMCell(
            input_size=hparams.attention_rnn_dim + hparams.encoder_embedding_dim,
            hidden_size=hparams.decoder_rnn_dim,
            bias=True,
        )

        self.linear_projection = LinearNorm(
            in_dim=hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            out_dim=hparams.n_mel_channels + hparams.n_frames_per_step,
        )

        self.gate_layer = LinearNorm(
            in_dim=hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            out_dim=1,
            bias=True,
            w_init_gain="sigmoid",
        )

    def get_go_frame(self, memory: Tensor) -> Tensor:
        B = memory.size(0)
        return torch.zeros(
            B, self.n_mel_channels * self.n_frames_per_step, device=memory.device
        )

    #! Check mel 3 dimension how?
    def parse_decoder_inputs(self, decoder_inputs: Tensor) -> Tensor:
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            decoder_inputs.size(1) // self.n_frames_per_step,
            -1,
        )
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        return decoder_inputs.transpose(0, 1)

    def initialize_decoder_states(self, memory, mask):
        B = memory.size(0)
        MAX_TIME = memory.size(1)
        device, dtype = memory.device, memory.dtype

        self.attention_hidden = torch.zeros(
            B, self.attention_rnn_dim, dtype=dtype, device=device
        )
        self.attention_cell = torch.zeros(
            B, self.attention_rnn_dim, dtype=dtype, device=device
        )
        self.decoder_hidden = torch.zeros(
            B, self.decoder_rnn_dim, dtype=dtype, device=device
        )
        self.decoder_cell = torch.zeros(
            B, self.decoder_rnn_dim, dtype=dtype, device=device
        )
        self.attention_weights = torch.zeros(B, MAX_TIME, dtype=dtype, device=device)
        self.attention_weights_cum = torch.zeros(
            B, MAX_TIME, dtype=dtype, device=device
        )
        self.attention_context = torch.zeros(
            B, self.encoder_embedding_dim, dtype=dtype, device=device
        )

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def decode(self, decoder_input):
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell)
        )
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training
        )

        attention_weights_cat = torch.cat(
            (
                self.attention_weights.unsqueeze(1),
                self.attention_weights_cum.unsqueeze(1),
            ),
            dim=1,
        )
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden,
            self.memory,
            self.processed_memory,
            attention_weights_cat,
            self.mask,
        )
        self.attention_weights_cum += self.attention_weights

        decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell)
        )
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training
        )

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1
        )
        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        return decoder_output, gate_prediction, self.attention_weights

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def forward(
        self, memory: Tensor, decoder_inputs: Tensor, memory_lengths: Tensor
    ) -> Tensor:
        init_decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((init_decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths)
        )

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory: Tensor) -> Tensor:
        init_decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )

        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step

        self.embedding = nn.Embedding(
            num_embeddings=hparams.n_symbols,
            embedding_dim=hparams.symbols_embedding_dim,
        )
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)

        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded),
        )

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.exapand(self.n_mel_channels, mask.size(0), mask.size(1))

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)

        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_length=text_lengths
        )

        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments], output_lengths
        )

    def inference(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_inputs, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)

        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]
        )
