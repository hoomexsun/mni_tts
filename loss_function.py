from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, model_output, targets):
        mel_target, gate_target, *_ = targets
        mel_out, mel_out_postnet, gate_out, *_ = model_output

        mel_target.requires_grad = False
        mel_loss = nn.MSELoss()(
            mel_out,
            mel_target,
        ) + nn.MSELoss()(
            mel_out_postnet,
            mel_target,
        )

        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)
        gate_out = gate_out.view(-1, 1)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        return mel_loss + gate_loss
