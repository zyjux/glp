from typing import Literal, Optional

import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2 as transforms


class glp_rotations(nn.Module):
    """Custom module implementing GLP rotation stacking

    Attributes
    ----------
    angles : list
        Angles the base image is rotated by.

    """

    def __init__(
        self,
        num_angles: int,
    ):
        """Constructor for glp_rotations class

        Arguments
        ----------
        num_angles : int
            Specifies the number of rotations to apply. Each rotation will be by 360 /
            num_angles degrees.
        """
        super().__init__()
        self.angles = np.linspace(0, 360, num=num_angles, endpoint=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Callable method for glp_rotations class. Rotates image and stacks the result

        Arguments
        ----------
        image : torch.Tensor
            Input image to be rotated.

        Returns
        ----------
        torch.Tensor
            Rotated versions of the input image stacked along the last dimension of the
            tensor. The image rotations are stacked in the order given in the angles
            attribute on this class (increasing, starting from 0 degrees).
        """
        if self.transform_type == "rotate":
            temp_list = [
                transforms.functional.rotate(image, angle) for angle in self.angles
            ]
        else:
            temp_list = [
                transforms.functional.affine(
                    image, angle=0, scale=scaling, translate=[0.0, 0.0], shear=[0.0]
                )
                for scaling in self.scalings
            ]
        return torch.stack(temp_list, dim=-1)


glp_rotations()


class glp_scalings(nn.Module):
    """
    Custom module implementing GLP rotation stacking


    """

    def __init__(
        self,
        transform_type: Literal["rotate", "scale"],
        num_angles: Optional[int] = None,
        scale_factor: Optional[float] = None,
        num_scalings: Optional[int] = None,
    ):
        """
        Constructor for glp_affine module


        args:
            transform_type ("rotate" or "scale"): Type of symmetry to apply to inputs.
            num_angles (int or None): Required if transform_type == "rotate". Specifies
                the number of rotations to apply. Each rotation will be by
                360 / num_angles degrees.
            scale_factor (float or None): Required if transform_type == "scale".
                Specifies the scaling proportion increment (both larger and smaller)
                used to build the list of scalings.

        """
        super().__init__()
        self.transform_type = transform_type
        if self.transform_type == "rotate":
            if num_angles is None:
                raise ValueError(
                    "num_angles must be specified for rotational transforms"
                )
            else:
                self.angles = np.linspace(0, 360, num=num_angles, endpoint=False)
        elif self.transform_type == "scale":
            if scale_factor is None or num_scalings is None:
                raise ValueError(
                    "scale_factor and num_scalings must be specified for scaling transforms"
                )
            else:
                self.scalings = np.linspace(
                    -scale_factor * num_scalings,
                    scale_factor * num_scalings,
                    2 * num_scalings + 1,
                )

    def forward(self, image):
        if self.transform_type == "rotate":
            temp_list = [
                transforms.functional.rotate(image, angle) for angle in self.angles
            ]
        else:
            temp_list = [
                transforms.functional.affine(
                    image, angle=0, scale=scaling, translate=[0.0, 0.0], shear=[0.0]
                )
                for scaling in self.scalings
            ]
        return torch.stack(temp_list, dim=-1)
