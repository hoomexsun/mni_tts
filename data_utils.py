import random
from typing import List, Tuple

from torch import Tensor
import numpy as np
import torch

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence


class TextMelLoader:

    def __init__(self, audiopaths_and_text: str, hparams) -> None:
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value: int = hparams.max_wav_value
        self.sr = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length,
            hparams.hop_length,
            hparams.win_length,
            hparams.n_mel_channels,
            hparams.sampling_rate,
            hparams.mel_fmin,
            hparams.mel_fmax,
        )
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(
        self, audiopath_and_text: Tuple[str, str]
    ) -> Tuple[Tensor, Tensor]:
        audio, text, *_ = audiopath_and_text
        return (self.get_text(text), self.get_mel(audio))

    def get_mel(self, filename) -> Tensor:
        if not self.load_mel_from_disk:
            waveform, sr = load_wav_to_torch(filename)
            if sr != self.stft.sampling_rate:
                raise ValueError(
                    f"{sr} SR doesn't match target {self.stft.sampling_rate} SR"
                )
            audio_norm = waveform / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert (
                melspec.size(0) == self.stft.n_mel_channels
            ), f"Mel dimension mismatch: Given {melspec.size(0)}, expected {self.stft.n_mel_channels}"
        return melspec

    def get_text(self, text) -> Tensor:
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self) -> int:
        return len(self.audiopaths_and_text)


class TextMelCollate:
    """
    Zero-pads model inputs (text) and targets (mel-spectrogram) based on the number
    of frames per step for batching during model training.

    Args:
        n_frames_per_step (int): Number of frames processed per step in the model.
    """

    def __init__(self, n_frames_per_step) -> None:
        self.n_frames_per_step = n_frames_per_step

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collates a batch of normalized text and mel-spectrograms by zero-padding
        them to the maximum sequence length in the batch for uniformity.

        Args:
            batch (List[Tuple[torch.Tensor, torch.Tensor]]):
                A list of tuples, where each tuple contains:
                - normalized text (torch.Tensor)
                - normalized mel-spectrogram (torch.Tensor)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - text_padded: Padded tensor of input texts
                - input_lengths: Lengths of input texts before padding
                - mel_padded: Padded tensor of mel-spectrograms
                - gate_padded: Gate tensor to indicate end of output
                - output_lengths: Lengths of mel-spectrograms before padding
        """
        # Sort the batch by the length of text sequences in descending order
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]  # Maximum length of text in the batch

        # Determine the number of mel channels and the maximum target length (mel-spectrogram)
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])

        # Ensure that the mel-spectrogram lengths are divisible by n_frames_per_step
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (
                self.n_frames_per_step - max_target_len % self.n_frames_per_step
            )
            assert max_target_len % self.n_frames_per_step == 0

        # Initialize tensors for padded text, mel, gate, and output lengths
        batch_size = len(batch)
        text_padded = torch.zeros(batch_size, max_input_len, dtype=torch.long)
        mel_padded = torch.zeros(
            batch_size, num_mels, max_target_len, dtype=torch.float
        )
        gate_padded = torch.zeros(batch_size, max_target_len, dtype=torch.float)
        output_lengths = torch.zeros(batch_size, dtype=torch.long)

        # Pad text and mel-spectrograms with zeros
        for idx, sorted_idx in enumerate(ids_sorted_decreasing):
            text, mel, *_ = batch[sorted_idx]
            text_padded[idx, : text.size(0)] = text
            mel_padded[idx, :, : mel.size(1)] = mel
            gate_padded[idx, mel.size(1) - 1 :] = 1  # Set remaining gate signal to 1
            output_lengths[idx] = mel.size(1)  # Store original length

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths
