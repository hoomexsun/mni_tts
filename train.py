from argparse import ArgumentParser
import os
import time

import torch

from config.hparams import HParam
from steps import (
    prepare_dataloaders,
    setup_logging_and_directories,
    load_model,
    warm_start_model,
    load_checkpoint,
    save_checkpoint,
    validate,
)
from loss_function import Tacotron2Loss


# Train function
def train(
    output_directory,
    log_directory,
    checkpoint_path,
    warm_start,
    hparams,
):
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    criterion = Tacotron2Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler() if hparams.fp16_run else None
    logger = setup_logging_and_directories(output_directory, log_directory)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    iteration = 0
    epoch_offset = 0
    lr = hparams.learning_rate

    # Load checkpoint if exists
    if checkpoint_path is not None:

        if warm_start:
            # Use warm start for transfer learning (fine-tuning from a pre-trained model)
            model = warm_start_model(checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _lr, _iteration = load_checkpoint(
                checkpoint_path,
                model,
                optimizer,
            )
            if hparams.use_saved_learning_rate:
                lr = _lr
            iteration = _iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    is_overflow = False

    # Training Loop
    for epoch in range(epoch_offset, hparams.epochs):
        print(f"Epoch: {epoch}")
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            model.zero_grad()

            x, y = model.parse_batch(batch)
            y_pred = model(x)

            loss = criterion(y_pred, y)
            reduced_loss = loss.item()

            if hparams.fp16_run:
                with torch.autocast(device_type="cuda"):
                    loss = criterion(y_pred, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh
                )
                scaler.step(optimizer)
                scaler.update()

                is_overflow = not torch.isfinite(grad_norm)
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh
                )
                optimizer.step()

            if not is_overflow:
                # Log
                duration = time.perf_counter() - start
                print(
                    f"Train loss {iteration} {reduced_loss:.6f} Grad Norm {grad_norm:.6f} {duration:.3f}s/it"
                )
                logger.log_training(reduced_loss, grad_norm, lr, duration, iteration)

                # Validate at checkpoint
                if iteration % hparams.iters_per_checkpoint == 0:
                    validate(
                        model,
                        criterion,
                        valset,
                        iteration,
                        hparams.batch_size,
                        collate_fn,
                        logger,
                    )
                    checkpoint_path = os.path.join(
                        output_directory, f"checkpoint_{iteration}"
                    )
                    save_checkpoint(model, optimizer, lr, iteration, checkpoint_path)

        # Next batch iteration
        iteration += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "-l",
        "--log_directory",
        type=str,
        help="directory to save tensorboard logs",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        default=None,
        required=False,
        help="checkpoint path",
    )
    parser.add_argument(
        "--warm_start",
        action="store_true",
        help="load model weights only, ignore specified layers",
    )
    parser.add_argument(
        "--hparams",
        type=str,
        default="config/default.yaml",
        required=False,
        help="path to the yaml config file",
    )

    args = parser.parse_args()
    # Create Hyper Parameter object
    hparams = HParam(args.hparams)

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    # Train on single GPU
    train(
        args.output_directory,
        args.log_directory,
        args.checkpoint_path,
        args.warm_start,
        hparams,
    )
