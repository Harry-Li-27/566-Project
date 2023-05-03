# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


def pixel_wised_loss(q, k, pixel_T=0.07):
    # normalize
    q = nn.functional.normalize(q, dim=1)
    k = nn.functional.normalize(k, dim=1)
    # gather all targets
    k = concat_all_gather(k)
    # add pixel-wised loss part, q is output of predictor, k is output of momentum_encoder part which dont update grad
    q_abs = q.norm(dim=1)
    k_abs = k.norm(dim=1)
    cos_similarity = torch.einsum('mc,nc->mn', q, k) / torch.einsum('i,j->ij', q_abs, k_abs)
    logits = cos_similarity / pixel_T
    N = logits.shape[0]
    labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
    return nn.CrossEntropyLoss()(logits, labels) * (2 * pixel_T)


def barlow_twins_loss(q, k, lamda=0.005):
    D = q.shape[1]
    N = q.shape[0]
    q_norm = (q - q.mean(0))/q.std(0)
    k_norm = (k - k.mean(0))/k.std(0)
    diagonal = torch.eye(D).cuda()
    c_m = torch.matmul(q_norm.T, k_norm)/N
    c_diff = (c_m - diagonal).pow(2)
    c_off_diagonal = torch.ones_like(c_diff)*lamda
    c_off_diagonal = c_off_diagonal.cuda()
    c_off_diagonal = c_off_diagonal - diagonal*(lamda-1)
    c_diff = c_diff*c_off_diagonal
    loss = c_diff.sum()
    return loss


class CombineLoss(nn.Module):
    def __init__(self):
        super(CombineLoss, self).__init__()
        self.beta = nn.Parameter(torch.tensor(1.0))  # Initialize the weight parameter

    def forward(self, loss1, loss2):
        # Apply the sigmoid function to alpha to ensure it's in the range (0, 1)
        weight = torch.sigmoid(self.beta)
        total_loss = weight * loss1 + (1 - weight) * loss2
        return total_loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

# Example usage
if __name__ == "__main__":
    criterion = CombineLoss()

    # Generate dummy losses
    loss1 = torch.tensor(1.0, requires_grad=True)
    loss2 = torch.tensor(2.0, requires_grad=True)

    # Calculate the total loss
    total_loss = criterion(loss1, loss2)
    print("Total loss:", total_loss.item())

    # Perform backward pass
    total_loss.backward()
    print("Gradients for loss1 and loss2:", loss1.grad, loss2.grad)