"""
Adversarial loss functions for GAN training.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..config import Config
from ..utils import get_logger

logger = get_logger()


class AdversarialLoss(nn.Module):
    """
    Base adversarial loss for GAN training.
    """

    def __init__(
            self,
            loss_type: str = "bce",
            target_real_label: float = 1.0,
            target_fake_label: float = 0.0,
            label_smoothing: bool = False,
            smoothing_factor: float = 0.1
    ):
        """
        Initialize adversarial loss.

        Args:
            loss_type: Type of loss ('bce', 'mse', 'hinge')
            target_real_label: Label for real images
            target_fake_label: Label for fake images
            label_smoothing: Whether to use label smoothing
            smoothing_factor: Smoothing factor for labels
        """
        super(AdversarialLoss, self).__init__()

        self.loss_type = loss_type.lower()
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        self.label_smoothing = label_smoothing
        self.smoothing_factor = smoothing_factor

        if self.loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif self.loss_type == "hinge":
            pass  # Hinge loss is implemented directly in forward
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        logger.debug(f"AdversarialLoss initialized: type={loss_type}, "
                     f"label_smoothing={label_smoothing}")

    def get_target_tensor(
            self,
            prediction: torch.Tensor,
            target_is_real: bool
    ) -> torch.Tensor:
        """
        Create target tensor with same type and device as prediction.

        Args:
            prediction: Discriminator predictions
            target_is_real: Whether target is real (True) or fake (False)

        Returns:
            Target tensor
        """
        # First, create the base target tensor using a scalar fill value.
        if target_is_real:
            target_tensor = torch.full_like(prediction, self.target_real_label)
            if self.label_smoothing:
                # Then, apply the smoothing operation to the new tensor.
                target_tensor -= self.smoothing_factor * torch.rand_like(prediction)
        else:
            target_tensor = torch.full_like(prediction, self.target_fake_label)
            if self.label_smoothing:
                # Apply smoothing to the tensor of zeros.
                target_tensor += self.smoothing_factor * torch.rand_like(prediction)

        return target_tensor

    def forward(
            self,
            prediction: torch.Tensor,
            target_is_real: bool
    ) -> torch.Tensor:
        """
        Compute adversarial loss.

        Args:
            prediction: Discriminator output logits
            target_is_real: Whether target should be real

        Returns:
            Adversarial loss
        """
        if self.loss_type == "hinge":
            if target_is_real:
                # Hinge loss for real: max(0, 1 - prediction)
                loss = torch.relu(1.0 - prediction).mean()
            else:
                # Hinge loss for fake: max(0, 1 + prediction)
                loss = torch.relu(1.0 + prediction).mean()
        else:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.criterion(prediction, target_tensor)

        return loss


class GeneratorLoss(nn.Module):
    """
    Generator loss combining content loss and adversarial loss.
    """

    def __init__(
            self,
            content_loss: nn.Module,
            adversarial_loss: AdversarialLoss,
            content_weight: float = 1.0,
            adversarial_weight: float = 0.001
    ):
        """
        Initialize generator loss.

        Args:
            content_loss: Content loss (MSE, L1, or Perceptual)
            adversarial_loss: Adversarial loss
            content_weight: Weight for content loss
            adversarial_weight: Weight for adversarial loss
        """
        super(GeneratorLoss, self).__init__()

        self.content_loss = content_loss
        self.adversarial_loss = adversarial_loss
        self.content_weight = content_weight
        self.adversarial_weight = adversarial_weight

        logger.debug(f"GeneratorLoss initialized: content_weight={content_weight}, "
                     f"adversarial_weight={adversarial_weight}")

    def forward(
            self,
            fake_images: torch.Tensor,
            real_images: torch.Tensor,
            discriminator_fake_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute generator loss.

        Args:
            fake_images: Generated images
            real_images: Real target images
            discriminator_fake_pred: Discriminator predictions for fake images

        Returns:
            Total loss and loss components dictionary
        """
        # Content loss
        content_loss_val = self.content_loss(fake_images, real_images)

        # Adversarial loss (generator wants discriminator to classify fakes as real)
        adversarial_loss_val = self.adversarial_loss(discriminator_fake_pred, target_is_real=True)

        # Total loss
        total_loss = (self.content_weight * content_loss_val +
                      self.adversarial_weight * adversarial_loss_val)

        # Loss components for logging
        loss_dict = {
            'g_content_loss': content_loss_val.detach(),
            'g_adversarial_loss': adversarial_loss_val.detach(),
            'g_total_loss': total_loss.detach()
        }

        return total_loss, loss_dict


class DiscriminatorLoss(nn.Module):
    """
    Discriminator loss for distinguishing real and fake images.
    """

    def __init__(self, adversarial_loss: AdversarialLoss):
        """
        Initialize discriminator loss.

        Args:
            adversarial_loss: Adversarial loss function
        """
        super(DiscriminatorLoss, self).__init__()
        self.adversarial_loss = adversarial_loss

        logger.debug("DiscriminatorLoss initialized")

    def forward(
            self,
            real_pred: torch.Tensor,
            fake_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute discriminator loss.

        Args:
            real_pred: Discriminator predictions for real images
            fake_pred: Discriminator predictions for fake images

        Returns:
            Total loss and loss components dictionary
        """
        # Loss for real images (should be classified as real)
        real_loss = self.adversarial_loss(real_pred, target_is_real=True)

        # Loss for fake images (should be classified as fake)
        fake_loss = self.adversarial_loss(fake_pred, target_is_real=False)

        # Total discriminator loss
        total_loss = (real_loss + fake_loss) * 0.5

        # Calculate accuracies for monitoring
        with torch.no_grad():
            real_acc = (torch.sigmoid(real_pred) > 0.5).float().mean()
            fake_acc = (torch.sigmoid(fake_pred) < 0.5).float().mean()

        # Loss components for logging
        loss_dict = {
            'd_real_loss': real_loss.detach(),
            'd_fake_loss': fake_loss.detach(),
            'd_total_loss': total_loss.detach(),
            'd_real_accuracy': real_acc,
            'd_fake_accuracy': fake_acc
        }

        return total_loss, loss_dict


class RelativisticAdversarialLoss(nn.Module):
    """
    Relativistic Adversarial Loss (RaGAN).
    The discriminator estimates the probability that real data is more realistic than fake data.
    """

    def __init__(self, loss_type: str = "bce"):
        """
        Initialize relativistic adversarial loss.

        Args:
            loss_type: Type of loss ('bce' or 'mse')
        """
        super(RelativisticAdversarialLoss, self).__init__()

        self.loss_type = loss_type.lower()

        if self.loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.loss_type == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

        logger.debug(f"RelativisticAdversarialLoss initialized: type={loss_type}")

    def forward(
            self,
            real_pred: torch.Tensor,
            fake_pred: torch.Tensor,
            for_discriminator: bool = True
    ) -> torch.Tensor:
        """
        Compute relativistic adversarial loss.

        Args:
            real_pred: Predictions for real images
            fake_pred: Predictions for fake images
            for_discriminator: Whether computing loss for discriminator or generator

        Returns:
            Relativistic adversarial loss
        """
        # Determine targets based on the network being trained
        if for_discriminator:
            # Discriminator wants to predict 1 for real and 0 for fake
            real_target = torch.ones_like(real_pred)
            fake_target = torch.zeros_like(fake_pred)
        else:
            # Generator wants the discriminator to predict 1 for fake and 0 for real
            real_target = torch.zeros_like(real_pred)
            fake_target = torch.ones_like(fake_pred)

        # Calculate relativistic predictions
        # For real images, compare to the average of fakes
        relativistic_real_pred = real_pred - fake_pred.mean()
        # For fake images, compare to the average of reals
        relativistic_fake_pred = fake_pred - real_pred.mean()

        # Calculate loss components
        real_loss = self.criterion(relativistic_real_pred, real_target)
        fake_loss = self.criterion(relativistic_fake_pred, fake_target)

        return (real_loss + fake_loss) * 0.5


def create_adversarial_losses_from_config(
        config: Config
) -> Tuple[Optional[GeneratorLoss], Optional[DiscriminatorLoss]]:
    """
    Create adversarial losses from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Generator loss and discriminator loss (None if not SRGAN)
    """
    if config.get('model', {}).get('name') != 'srgan':
        return None, None

    # Import here to avoid circular imports
    from .pixel_loss import create_pixel_loss_from_config
    from .perceptual_loss import create_perceptual_loss_from_config

    srgan_config = config.get('srgan', {})
    loss_config = srgan_config.get('loss', {})
    smoothing_config = loss_config.get('label_smoothing', {})

    logger.info("Creating adversarial losses from configuration...")

    # Create content loss
    content_loss_type = loss_config.get('content_loss', 'mse')
    if content_loss_type == 'vgg':
        content_loss = create_perceptual_loss_from_config(config)
        if content_loss is None:
            raise ValueError("Failed to create VGG perceptual loss from config.")
    else:
        content_loss = create_pixel_loss_from_config(config)

    # Create base adversarial loss
    adversarial_loss = AdversarialLoss(
        loss_type='bce',
        label_smoothing=smoothing_config.get('enabled', False),
        smoothing_factor=smoothing_config.get('factor', 0.1)
    )

    # Loss weights
    content_weight = loss_config.get('content_weight', 1.0)
    adversarial_weight = loss_config.get('adversarial_weight', 0.001)

    # Create composite generator and discriminator losses
    generator_loss = GeneratorLoss(
        content_loss=content_loss,
        adversarial_loss=adversarial_loss,
        content_weight=content_weight,
        adversarial_weight=adversarial_weight
    )

    discriminator_loss = DiscriminatorLoss(adversarial_loss)

    logger.info("Generator and Discriminator loss functions created successfully.")

    return generator_loss, discriminator_loss
