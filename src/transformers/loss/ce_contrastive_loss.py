"""

"""

import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F

class TripletLoss(nn.Module):
    def __init__(
            self,
            reduction: str
    ) -> None:
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(reduction=reduction)

    def forward(self, anchors: Tensor, positives: Tensor, negatives:Tensor) -> Tensor:
        # Compute loss
        loss = self.triplet_loss(anchor=anchors, positive=positives, negative=negatives)
        return loss


class CrossEntropyAndTripletLoss(nn.Module):
    def __init__(
            self,
            caption_loss_weight,
            triplet_loss_weight,
    ):
        super().__init__()

        self.caption_loss_weight = caption_loss_weight
        self.triplet_loss_weight = triplet_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(reduction="mean")
        self.triplet_loss = TripletLoss(reduction="mean")

    def forward(self, logits, labels, output_dict=False):
        """
            B: batch
            V: number of vocabulary tokens
            S: sentence (tokens)

            Args:
                logits: model prediction [B x S x V ]
                labels: considered tokens sampled [B x S ]
        """
        # Get only anchor negative examples to compute captioning loss
        mask = torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
        mask[1::3] = False
        anchor_negatives_logits = logits[mask, :, :]
        anchor_negatives_labels = labels[mask, :]
        caption_loss = self.caption_loss(
            anchor_negatives_logits.permute(0, 2, 1),
            anchor_negatives_labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        # Separate into anchors, positives and negatives
        mask = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
        mask[::3] = True
        anchors = logits[mask, :, :]
        mask = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
        mask[1::3] = True
        positives = logits[mask, :, :]
        mask = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
        mask[2::3] = True
        negatives = logits[mask, :, :]
        triplet_loss = self.triplet_loss(
            anchors=anchors,
            positives=positives,
            negatives=negatives
        )
        triplet_loss = triplet_loss * self.triplet_loss_weight

        if output_dict:
            return {"caption_loss": caption_loss, "triplet_loss": triplet_loss}

        return caption_loss, triplet_loss
