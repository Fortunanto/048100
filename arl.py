###############################################################################
# MIT License
#
# Copyright (c) 2020 Jardenna Mohazzab, Luc Weytingh, 
#                    Casper Wortmann, Barbara Brocades Zaalberg
#
# This file contains an implementation of the ARL model prented in "Fairness 
# without Demographics through Adversarially Reweighted Learning" by Lahoti 
# et al..
#
# Author: Jardenna Mohazzab, Luc Weytingh, 
#         Casper Wortmann, Barbara Brocades Zaalberg 
# Date Created: 2021-01-01
###############################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights


class LearnerNN(nn.Module):
    def __init__(
        self, device='cpu'
    ):
        """
        Implements the learner DNN.
        Args:
          embedding_size: list of tuples (n_classes, n_features) containing
                           embedding sizes for categorical columns.
          n_num_cols: number of numerical inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer.
          activation_fn: the activation function to use.
        """
        super().__init__()
        self.device = device

        self.model = models.resnet50(weights=ResNet50_Weights)
        # We want to fine tune the last layer
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        """
        The forward step for the learner.
        """

        x = self.model(x)
        return x


class AdversaryNN(nn.Module):
    def __init__(self, device='cpu'):
        """
        Implements the adversary DNN.
        Args:
          embedding_size: list of tuples (n_classes, n_features) containing
                          embedding sizes for categorical columns.
          n_num_cols: number of numerical inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer.
        """
        super().__init__()
        self.device = device

        self.model = models.resnet50(weights=ResNet50_Weights)
        # We want to fine tune the last layer
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        The forward step for the adversary.
        """
        x = self.model(x)

        x = self.sigmoid(x)
        x_mean = torch.mean(x)
        x = x / torch.max(torch.Tensor([x_mean, 1e-4]))
        x = x + torch.ones_like(x)

        return x


class ARL(nn.Module):

    def __init__(
        self,
        batch_size=256,
        device='cuda',
    ):
        """
        Combines the Learner and Adversary into a single module.

        Args:
          embedding_size: list of tuples (n_classes, embedding_dim) containing
                    embedding sizes for categorical columns.
          n_num_cols: the amount of numerical columns in the data.
          learner_hidden_units: list of ints, specifies the number of units
                    in each linear layer for the learner.
          adversary_hidden_units: list of ints, specifies the number of units
                    in each linear layer for the learner.
          batch_size: the batch size.
          activation_fn: the activation function to use for the learner.
        """
        super().__init__()
        torch.autograd.set_detect_anomaly(True)

        self.device = device
        self.adversary_weights = torch.ones(batch_size, 1)

        self.learner = LearnerNN(
            device=device
        )
        self.adversary = AdversaryNN(
            device=device
        )

        self.learner.to(device)
        self.adversary.to(device)


    def learner_step(self, x, targets):
        self.learner.zero_grad()
        prediction = self.learner(x)

        # Logits are used for calculating loss
        adversary_weights = self.adversary_weights.to(self.device)
        loss = self.get_learner_loss(prediction, targets, adversary_weights)
        loss.backward()

        # Predictions are returned to trainer for fairness metrics
        logging_dict = {"learner_loss": loss}
        return loss, prediction, logging_dict

    def adversary_step(self, x, learner_logits, targets):
        """
        Performs one loop
        """
        self.adversary.zero_grad()

        adversary_weights = self.adversary(x)
        self.adversary_weights = adversary_weights.detach()

        loss = self.get_adversary_loss(
            learner_logits.detach(), targets, adversary_weights
        )

        loss.backward()

        logging_dict = {"adv_loss": loss}
        return loss, logging_dict

    def get_learner_loss(self, prediction, targets, adversary_weights):
        """
        Compute the loss for the learner.
        """
        loss = F.cross_entropy(prediction,targets,reduction="none")
        loss.to(self.device)
        
        weighted_loss = loss * adversary_weights
        weighted_loss = torch.mean(weighted_loss)
        return weighted_loss

    def get_adversary_loss(self, prediction, targets, adversary_weights):
        """
        Compute the loss for the adversary.
        """
        loss = F.cross_entropy(prediction,targets,reduction="none")
        loss.to(self.device)

        weighted_loss = -(adversary_weights * loss)
        weighted_loss = torch.mean(weighted_loss)
        return weighted_loss
