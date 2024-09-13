import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from data_utils import TextMelLoader, TextMelCollate
from logger import Tacotron2Logger
from model import Tacotron2


def prepare_dataloaders(
    hparams: dict,
) -> Tuple[DataLoader, TextMelLoader, TextMelCollate]:
    train_set = TextMelLoader(hparams.training_files, hparams)
    val_set = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=hparams.batch_size,
        shuffle=True,  # Use False for multi GPU
        sampler=None,  # Use torch.utils.data.distributed.DistributedSampler for multi GPU
        num_workers=1,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    return train_loader, val_set, collate_fn


def setup_logging_and_directories(output_directory, log_directory) -> Tacotron2Logger:
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o775)
    return Tacotron2Logger(os.path.join(output_directory, log_directory))


def load_model(hparams: dict) -> Tacotron2:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Tacotron2(hparams).to(device)
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = torch.finfo("float16").min
    return model


def warm_start_model(
    checkpoint_path: str, model: Tacotron2, ignore_layers: List[str] = []
) -> Tacotron2:
    assert os.path.isfile(checkpoint_path)
    print(f"Warm starting model from checkpoint '{checkpoint_path}'")
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model_dict = checkpoint_dict.get("state_dict", {})
    if ignore_layers:
        model_dict = {k: v for k, v in model_dict.items() if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict, strict=False)  # Allow for missing keys if any
    return model


def load_checkpoint(
    checkpoint_path: str, model: Tacotron2, optimizer: torch.optim.Optimizer
) -> tuple:
    assert os.path.isfile(checkpoint_path)
    print(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    lr = checkpoint_dict["learning_rate"]
    iteration = checkpoint_dict["iteration"]
    print(f"Loaded checkpoint '{checkpoint_path}' from iteration {iteration}")
    return model, optimizer, lr, iteration


def save_checkpoint(
    model: Tacotron2,
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    iteration: int,
    filepath: str,
) -> None:
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    torch.save(
        {
            "iteration": iteration,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        filepath,
    )


def validate(
    model,
    criterion,
    valset,
    iteration,
    batch_size,
    collate_fn,
    logger: Tacotron2Logger,
):
    model.eval()
    with torch.inference_mode():
        val_loader = DataLoader(
            dataset=valset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn,
            pin_memory=False,
        )
        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    print(f"Validation loss {iteration}: {val_loss:9f}  ")
    logger.log_validation(val_loss, model, y, y_pred, iteration)
