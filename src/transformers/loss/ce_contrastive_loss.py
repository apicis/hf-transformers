""" Implementation of the combination between crossentropy and triplet loss. 
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

    def forward(self, logits, labels, latent_vector, output_dict=False):
        """
            B: batch
            V: number of vocabulary tokens
            S: sentence (tokens)
            E: embedding

            Args:
                logits: model prediction [B x S x V ]
                labels: considered tokens sampled [B x S ]
                latent_vector: latent vector to compute the triplet loss with [B x E]
        """
        # Get only anchor examples to compute captioning loss
        mask = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
        mask[::3] = True
        anchor_logits = logits[mask, :, :]
        anchor_labels = labels[mask, :]
        caption_loss = self.caption_loss(
            anchor_logits.permute(0, 2, 1),
            anchor_labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        # Separate into anchors, positives and negatives
        mask = torch.zeros(latent_vector.shape[0], dtype=torch.bool, device=latent_vector.device)
        mask[::3] = True
        anchors = latent_vector[mask]
        mask = torch.zeros(latent_vector.shape[0], dtype=torch.bool, device=latent_vector.device)
        mask[1::3] = True
        positives = latent_vector[mask]
        mask = torch.zeros(latent_vector.shape[0], dtype=torch.bool, device=latent_vector.device)
        mask[2::3] = True
        negatives = latent_vector[mask]
        triplet_loss = self.triplet_loss(
            anchors=anchors,
            positives=positives,
            negatives=negatives
        )
        triplet_loss = triplet_loss * self.triplet_loss_weight

        if output_dict:
            return {"caption_loss": caption_loss, "triplet_loss": triplet_loss}

        return caption_loss, triplet_loss
