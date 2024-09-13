from typing import List, Tuple


from torch import Tensor
import torch
from torchaudio import load


# # MINE
# def load_data(file, split="\t"):
#     with open(file, encoding="utf-8") as f:
#         data = [line.strip().split(split) for line in f]
#     return data


# def read_list(file):
#     with open(file, encoding="utf-8") as f:
#         list_ = [line.strip() for line in f if line.strip()]
#     return list_


def load_wav_to_torch(wav_path: str) -> Tuple[Tensor, int]:
    waveform, sr = load(wav_path)
    return waveform.squeeze(0), sr


def load_filepaths_and_text(filename, split="|") -> List[Tuple[str, str]]:
    with open(filename, encoding="utf-8") as f:
        data = [tuple(line.strip().split(split)) for line in f]
    return data


def get_mask_from_lengths(lengths: Tensor) -> Tensor:
    max_len = lengths.max().item()
    ids = torch.arange(max_len, device=lengths.device, dtype=torch.long)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    # return torch.autograd.Variable(x)
    return x
