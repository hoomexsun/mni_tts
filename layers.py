import librosa
from torch import Tensor, nn
import torch

from audio_processing import dynamic_range_compression, dynamic_range_decompression
from stft import STFT


class LinearNorm(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        w_init_gain="linear",
    ) -> None:
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        # Xavier uniform initialization for better convergence
        nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_layer(x)


class ConvNorm(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        bias: bool = True,
        w_init_gain="linear",
    ) -> None:
        super().__init__()
        if padding is None:
            assert (
                kernel_size % 2 == 1
            ), "Kernel size must be odd if padding is not specified"
            padding = dilation * (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, signal: Tensor) -> Tensor:
        return self.conv(signal)


class TacotronSTFT(nn.Module):
    """
    Computes STFT and mel-spectrograms from input waveforms.

    Args:
        filter_length (int): Number of fft bins.
        hop_length (int): Hop size for STFT.
        win_length (int): Window length for STFT.
        n_mel_channels (int): Number of mel filter banks.
        sampling_rate (int): Audio sample rate.
        mel_fmin (float): Minimum frequency for mel filter bank.
        mel_fmax (float): Maximum frequency for mel filter bank.
    """

    def __init__(
        self,
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mel_channels: int = 80,
        sampling_rate: int = 22050,
        mel_fmin: float = 0.0,
        mel_fmax: float = 8000.0,
    ) -> None:
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)

        # Generate mel filter banks using librosa
        mel_basis = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=filter_length,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )

        # Register mel basis as a non-trainable buffer (ensures it is saved with the model)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())

    def spectral_normalize(self, magnitudes: Tensor) -> Tensor:
        return dynamic_range_compression(magnitudes)

    def spectral_denormalize(self, magnitudes: Tensor) -> Tensor:
        return dynamic_range_decompression(magnitudes)

    def mel_spectrogram(self, y: Tensor) -> Tensor:
        assert torch.min(y.data) >= -1, "Waveform should be >= -1"
        assert torch.max(y.data) <= 1, "Waveform should be <= 1"

        # Compute STFT
        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data

        # Apply mel filter banks
        mel_output = torch.matmul(self.mel_basis, magnitudes)

        # Normalize the mel-spectrogram
        mel_output = self.spectral_normalize(mel_output)

        return mel_output
