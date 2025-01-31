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


class NegativeLearningLossRandomSample(nn.Module):
    def __init__(
            self,
            reduction: str,
            tokens_num: int,
            token_factor: int
    ) -> None:
        super().__init__()
        self.tokens_num = tokens_num
        self.token_factor = token_factor
        self.nll_loss = nn.NLLLoss(reduction=reduction)

    def get_probabilities_considered_tokens(self, model_probs: Tensor, considered_tokens: Tensor) -> Tensor:
        """ Selects the probabilities of the considered tokens

            B: batch
            V: number of vocabulary tokens
            S: sentence (tokens)
            T: number of sampled tokens

        Args:
            model_probs: probabilities of the model prediction [B x S x V ]
            considered_tokens: considered tokens sampled [B x S x T ]

        Returns:
            probs: probabilities of the considered tokens [B x S x T ]
        """
        # Get considered tokens shape
        b, s, tokens_num = considered_tokens.shape

        # Create mask in considered token indices
        considered_tokens_mask = torch.zeros_like(model_probs, device=model_probs.device, dtype=torch.int64).scatter_(
            -1,
            considered_tokens,
            1.) > 0
        # Extract probabilities of the considered tokens
        probs = model_probs[considered_tokens_mask].view(b, s, tokens_num)
        return probs

    def randomly_sample_from_set(self, tokens_num: int, tokens_set: Tensor) -> Tensor:
        """ Randomly sample tokens_num tokens from tokens_set

            V: number of vocabulary tokens
            S: sentence (tokens)
            T: number of tokens to sample

        Args:
            tokens_num: number of tokens to randomly sample
            tokens_set: set of tokens to sample randomly from [S x V]

        Returns:
            random_tokens: tokens sampled randomly [S x T]
        """
        # Get shape of the token set
        s, v = tokens_set.shape

        # Randomly sample the tokens from the token_set
        random_tokens_ind = torch.zeros(size=[s, tokens_num], dtype=torch.int64, device=tokens_set.device)
        src_tensor = torch.ones_like(tokens_set)
        for token in range(s):
            random_tokens_ind[token] = torch.multinomial(torch.ones(v), num_samples=tokens_num, replacement=False)

        # Create random tokens mask
        random_tokens_mask = torch.zeros_like(tokens_set, device=tokens_set.device).scatter_(dim=-1,
                                                                                             index=random_tokens_ind,
                                                                                             src=src_tensor) > 0

        # Extract tokens from token set
        random_tokens = tokens_set[random_tokens_mask].view(s, tokens_num)
        return random_tokens

    def sample_unshared_tokens_descending(self, model_logits: Tensor, target_tokens: Tensor, tokens_num: int,
                                          token_factor: int) -> Tensor:
        """ Samples a set of tokens_num from the set of tokens_num*tokens_factor that have highest logit values (hence probability).

            B: batch
            V: number of vocabulary tokens
            S: sentence (tokens)
            T: number of tokens to sample

        Args:
            model_logits: logits of the model prediction [B x S x V ]
            target_tokens: tokens in the target tensor [B x S]
            tokens_num: number of tokens to randomly sample
            token_factor: multiplying factor of random tokens to enlarge the set to sample from

        Returns:
            sampled_tokens: tokens randomly sampled among the tokens with highest logit values [B x S x T]
        """
        # Get logits shape
        b, s, v = model_logits.shape

        # Initialize tensors
        sampled_tokens = torch.zeros([b, s, tokens_num], dtype=torch.int64, device=model_logits.device)
        logits_temp = model_logits.clone()

        for batch in range(b):
            # Get unique tokens in target
            unique_target_tokens = torch.unique(target_tokens[batch].flatten())

            # Create the mask of target tokens
            target_mask = torch.zeros(size=[s, v], dtype=torch.bool, device=model_logits.device)
            target_mask[:, unique_target_tokens] = True

            # Discard logits of token mask
            logits_temp[batch, target_mask] = -torch.inf

            # Sort logits decreasing and get first indices (tokens)
            desc_tokens_subset = torch.sort(logits_temp[batch], dim=-1, descending=True)[1][:,
                                 0:tokens_num * token_factor]

            # Randomly sample from the set of tokens having highest logit values
            sampled_tokens[batch] = self.randomly_sample_from_set(tokens_num, tokens_set=desc_tokens_subset)
        return sampled_tokens

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        # Get probabilities
        model_probs = F.softmax(inputs, dim=-1)

        # Sample token randomly
        most_probable_tokens = self.sample_unshared_tokens_descending(model_logits=inputs,
                                                                      target_tokens=targets,
                                                                      tokens_num=self.tokens_num,
                                                                      token_factor=self.token_factor)

        # Get probabilities for those tokens
        sampled_probs = self.get_probabilities_considered_tokens(model_probs, most_probable_tokens)

        # Compute loss for random token
        loss_random = - torch.sum(torch.log(1 - sampled_probs))

        return loss_random


class PositiveNegativeTripletLoss(nn.Module):
    def __init__(
            self,
            caption_loss_weight,
            negative_loss_weight,
            triplet_loss_weight
    ):
        super().__init__()

        self.caption_loss_weight = caption_loss_weight
        self.negative_loss_weight = negative_loss_weight
        self.triplet_loss_weight = triplet_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(reduction="mean")
        self.negative_loss = NegativeLearningLossRandomSample(reduction="mean", tokens_num=1000, token_factor=32)
        self.triplet_loss = TripletLoss(reduction="mean")

    def forward(self, logits, labels, output_dict=False):
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

        negative_loss = self.negative_loss(
            anchor_negatives_logits,
            anchor_negatives_labels,
        )
        negative_loss = negative_loss * self.negative_loss_weight

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
            return {"caption_loss": caption_loss, "negative_loss": negative_loss, "triplet_loss": triplet_loss}

        return caption_loss, negative_loss, triplet_loss
