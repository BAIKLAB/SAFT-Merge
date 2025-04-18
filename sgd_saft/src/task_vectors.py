from collections import OrderedDict
import os
import pickle
from typing import Dict, Optional
from typing_extensions import Self

import torch
from torch import nn


class TaskVector:
    def __init__(
        self, 
        pretrained_checkpoint: Optional[os.PathLike] = None, 
        finetuned_checkpoint: Optional[os.PathLike] = None, 
        vector: Optional[Self] = None,
        is_cuda: bool = True,
    ):
        self.is_cuda = is_cuda
        if vector is not None:
            self.vector = vector
        else:
            assert (
                pretrained_checkpoint is not None and finetuned_checkpoint is not None
            )

            pretrained_state_dict = self._load_checkpoint(pretrained_checkpoint)
            if isinstance(pretrained_state_dict, nn.Module):
                pretrained_state_dict = pretrained_state_dict.state_dict()
            finetuned_state_dict = self._load_checkpoint(finetuned_checkpoint)
            if isinstance(finetuned_state_dict, nn.Module):
                finetuned_state_dict = finetuned_state_dict.state_dict()
            
            self.vector = OrderedDict()
            with torch.no_grad():
                # for key in param_names_to_merge:
                for key in pretrained_state_dict.keys():
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue

                    if self.is_cuda:
                        pretrained_state_dict[key] = pretrained_state_dict[key]
                        finetuned_state_dict[key] = finetuned_state_dict[key]

                    self.vector[key] = (
                        finetuned_state_dict[key] - pretrained_state_dict[key]
                    )

    def _load_checkpoint(self, checkpoint: os.PathLike) -> nn.Module:
        """Load a checkpoint into a model."""
        try:
            return torch.load(checkpoint)
        except RuntimeError:
            return pickle.load(open(checkpoint, 'rb'))

    def __add__(self, other: Self) -> Self:
        """Add two task vectors together."""
        assert isinstance(other, TaskVector), "addition of TaskVector can only be done with another TaskVector!"
        new_vector = OrderedDict()
        with torch.no_grad():
            for key in self.vector.keys():
                assert key in other.vector.keys(), f"param_name {key} is not contained in both task vectors!"
                new_vector[key] = self.vector[key] + other.vector[key]
        return self.__class__(vector=new_vector)

    def __sub__(self, other: Self) -> Self:
        """Subtract two task vectors."""
        return self.__add__(-other)

    def __radd__(self, other: Self) -> Self:
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self) -> Self:
        """Negate a task vector."""
        new_vector = OrderedDict()
        with torch.no_grad():
            for key in self.vector.keys():
                new_vector[key] = -self.vector[key]
        return self.__class__(vector=new_vector)

    def __pow__(self, power: int) -> Self:
        """Power of a task vector."""
        new_vector = OrderedDict()
        with torch.no_grad():
            for key in self.vector.keys():
                new_vector[key] = self.vector[key] ** power
        return self.__class__(vector=new_vector)

    def __mul__(self, other: float) -> Self:
        """Multiply a task vector by a scalar."""
        new_vector = OrderedDict()
        with torch.no_grad():
            for key in self.vector.keys():
                new_vector[key] = other * self.vector[key]
        return self.__class__(vector=new_vector)

    def dot(self, other: Self) -> torch.Tensor:
        """Dot product of two task vectors."""
        other = self._cast_to_same_type(other)
        with torch.no_grad():
            dot_product = 0.0
            for key in self.vector.keys():
                assert key in other.vector.keys(), f"param_name {key} is not contained in both task vectors!"
                dot_product += torch.sum(self.vector[key] * other.vector[key])
        return dot_product

    def norm(self) -> torch.Tensor:
        """Norm of a task vector."""
        return torch.sqrt(self.dot(self))

    def apply_to(self, pretrained_checkpoint: os.PathLike, scaling_coef: float = 1.0) -> Dict[str, torch.Tensor]:
        """Apply a task vector to a pretrained model."""
        pretrained_state_dict = self._load_checkpoint(pretrained_checkpoint)
        if isinstance(pretrained_state_dict, nn.Module):
            pretrained_state_dict = pretrained_state_dict.state_dict()

        new_state_dict = OrderedDict()
        with torch.no_grad():
            for key in self.vector.keys():
                if self.is_cuda:
                    pretrained_state_dict[key] = pretrained_state_dict[key]
                    
                new_state_dict[key] = (
                    pretrained_state_dict[key] + scaling_coef * self.vector[key]
                )
        return new_state_dict