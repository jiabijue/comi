import numpy as np
import torch
import torch.nn.functional as F


class TokenSampler:
    def __init__(self, model, device, tgt_len=1, mem_len=896, ext_len=0):
        if tgt_len != 1:
            raise ValueError()
        if ext_len != 0:
            raise ValueError()
        self.model = model
        self.model.eval()
        self.model.reset_length(1, ext_len, mem_len)
        self.device = device
        self.reset()

    def reset(self):
        self.mems = []
        self.generated = []

    @torch.no_grad()
    def sample_next_token_updating_mem(self, last_token, temp=1., topk=None):

        # Sanitize sampling params
        if temp < 0:
            raise ValueError()
        if topk is not None and topk < 1:
            raise ValueError()

        # Append last input token because we've officially selected it
        self.generated.append(last_token)

        # Create input array
        _inp = [last_token]
        _inp = np.array(_inp, dtype=np.int64)[:, np.newaxis]
        inp = torch.from_numpy(_inp).to(self.device)

        # Evaluate the model, saving its memory.
        ret = self.model.forward_generate(inp, *self.mems)
        all_logits, self.mems = ret[0], ret[1:]

        # Select last timestep, only batch item
        logits = all_logits[-1, 0]

        # Handle temp 0 (argmax) case
        if temp == 0:
            probs = torch.zeros_like(logits)
            probs[logits.argmax()] = 1.
        else:
            # Apply temperature spec
            if temp != 1:
                logits /= temp

            # Compute softmax
            probs = F.softmax(logits, dim=-1)

        # Select top-k if specified
        if topk is not None:
            _, top_idx = torch.topk(probs, topk)
            mask = torch.zeros_like(probs)
            mask[top_idx] = 1.
            probs *= mask
            probs /= probs.sum()

        # Sample from probabilities
        token = torch.multinomial(probs, 1)
        token = int(token.item())

        return token, probs