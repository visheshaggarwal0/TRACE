import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSkeletonize(nn.Module):
    """Differentiable soft-skeletonization using morphological min/max pooling.

    Implements the differentiable mathematical formulation of morphological
    skeletonization (Shit et al., CVPR 2021).
    """

    def __init__(self, num_iters: int = 10):
        super().__init__()
        self.num_iters = num_iters

    def soft_erode(self, img: torch.Tensor) -> torch.Tensor:
        if len(img.shape) != 4:
            raise ValueError("Input tensor must have shape (B, C, H, W)")
        p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
        return torch.min(p1, p2)

    def soft_dilate(self, img: torch.Tensor) -> torch.Tensor:
        if len(img.shape) != 4:
            raise ValueError("Input tensor must have shape (B, C, H, W)")
        p1 = F.max_pool2d(img, (3, 1), stride=1, padding=(1, 0))
        p2 = F.max_pool2d(img, (1, 3), stride=1, padding=(0, 1))
        return torch.max(p1, p2)

    def soft_open(self, img: torch.Tensor) -> torch.Tensor:
        return self.soft_dilate(self.soft_erode(img))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Computes the soft skeleton of a continuous probability map [0, 1]."""
        img = torch.clamp(img, 0.0, 1.0)
        skel = F.relu(img - self.soft_open(img))
        for _ in range(self.num_iters):
            img = self.soft_erode(img)
            skel = skel + F.relu(img - self.soft_open(img))
        return torch.clamp(skel, 0.0, 1.0)


def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Computes soft Dice loss for class-imbalanced segmentation."""
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(-2, -1))
    denominator = (pred + target).sum(dim=(-2, -1))
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def soft_cldice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    skeletonizer: SoftSkeletonize,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Computes differentiable Centerline Dice (clDice) loss to preserve road connectivity."""
    prob = torch.sigmoid(logits)
    skel_pred = skeletonizer(prob)
    skel_true = skeletonizer(target)

    # Topological Precision: Skeleton of prediction covered by true mask
    t_prec = (skel_pred * target).sum(dim=(-2, -1)) + smooth
    t_prec_denom = skel_pred.sum(dim=(-2, -1)) + smooth
    prec = t_prec / t_prec_denom

    # Topological Sensitivity: True skeleton covered by prediction
    t_sens = (prob * skel_true).sum(dim=(-2, -1)) + smooth
    t_sens_denom = skel_true.sum(dim=(-2, -1)) + smooth
    sens = t_sens / t_sens_denom

    cldice = (2.0 * prec * sens) / (prec + sens)
    return 1.0 - cldice.mean()


class TRACECompositeLoss(nn.Module):
    """Composite Multi-Objective Loss: BCE + Dice + clDice.

    Balances pixel-wise probability gradients, class imbalance, and
    topological centerline connectivity.
    """

    def __init__(
        self,
        lambda_bce: float = 1.0,
        lambda_dice: float = 1.0,
        lambda_cldice: float = 1.5,
        cldice_iters: int = 8,
    ):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_dice = lambda_dice
        self.lambda_cldice = lambda_cldice
        self.bce = nn.BCEWithLogitsLoss()
        self.skeletonizer = SoftSkeletonize(num_iters=cldice_iters)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(logits, targets)
        loss_dice = soft_dice_loss(logits, targets)
        loss_cldice = soft_cldice_loss(logits, targets, self.skeletonizer)

        total_loss = (
            self.lambda_bce * loss_bce
            + self.lambda_dice * loss_dice
            + self.lambda_cldice * loss_cldice
        )
        return total_loss
