import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        greedy_tokens = logits.argmax(dim=-1)
        safe_temperatures = temperatures.clamp_min(1e-10).unsqueeze(dim=1)
        probs = torch.softmax(logits / safe_temperatures, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
