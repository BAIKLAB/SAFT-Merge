import abc
from collections import OrderedDict
import os
from typing import Dict, Optional, Sequence, Union
from typing_extensions import Self

import torch
from torch import nn

from linearize import LinearizedImageEncoder


class _TaskVector(abc.ABC):
    def __init__(
        self, 
        pretrained_checkpoint: Optional[os.PathLike] = None, 
        finetuned_checkpoint: Optional[os.PathLike] = None, 
        vector: Optional[Self] = None,
    ):
        assert (
            pretrained_checkpoint is not None and finetuned_checkpoint is not None
        )

        pretrained_state_dict = self._load_checkpoint(pretrained_checkpoint)
        if isinstance(pretrained_state_dict, nn.Module):
            pretrained_state_dict = pretrained_state_dict.state_dict()
        pretrained_state_dict.pop('model_name', None)
        self.pretrained_state_dict = pretrained_state_dict
        
        finetuned_state_dict = self._load_checkpoint(finetuned_checkpoint)
        if isinstance(finetuned_state_dict, nn.Module):
            finetuned_state_dict = finetuned_state_dict.state_dict()
        finetuned_state_dict.pop('model_name', None)
        self.finetuned_state_dict = finetuned_state_dict

    @abc.abstractmethod
    def _load_checkpoint(self, checkpoint: os.PathLike) -> Union[nn.Module, Dict[str, torch.Tensor]]:
        """Load a checkpoint into a model."""
        raise NotImplementedError

    @abc.abstractmethod
    def _cast_to_same_type(self, other: Self) -> Self:
        raise NotImplementedError

    def __add__(self, other: Self) -> Self:
        """Add two task vectors together."""
        assert isinstance(other, _TaskVector), "addition of TaskVector can only be done with another TaskVector!"
        other = self._cast_to_same_type(other)
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

    def __mul__(self, other: Self) -> Self:
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
                new_state_dict[key] = (
                    pretrained_state_dict[key] + scaling_coef * self.vector[key]
                )
        return new_state_dict


class NonLinearTaskVector(_TaskVector):
    """A task vector for nonlinear models."""
    def __init__(
        self, 
        pretrained_checkpoint: Optional[os.PathLike] = None, 
        finetuned_checkpoint: Optional[os.PathLike] = None, 
        vector: Optional[Self] = None,
    ):
        if vector is not None:
            self.vector = vector
        else:
            super().__init__(
                pretrained_checkpoint,
                finetuned_checkpoint,
                vector,
            )
            
            self.vector = OrderedDict()
            with torch.no_grad():
                for key in self.finetuned_state_dict.keys():
                    pt_key = key.replace('image_encoder.', '')
                    if self.pretrained_state_dict[pt_key].dtype in [torch.int64, torch.uint8]:
                        continue

                    self.vector[key] = (
                        self.finetuned_state_dict[key] - self.pretrained_state_dict[pt_key]
                    )

    def _load_checkpoint(self, checkpoint: os.PathLike) -> Union[nn.Module, Dict[str, torch.Tensor]]:
        """Load a checkpoint into a model."""
        return torch.load(checkpoint, map_location="cpu")

    def apply_to_nonlinear(self, pretrained_nonlinear_checkpoint: os.PathLike, scaling_coef: float = 1.0) -> Dict[str, torch.Tensor]:
        """Apply a task vector to a nonlinear pretrained model."""
        return self.apply_to(pretrained_nonlinear_checkpoint, scaling_coef)

    def apply_to_linear(self, pretrained_linear_checkpoint: os.PathLike, scaling_coef: float = 1.0) -> Dict[str, torch.Tensor]:
        """Apply a task vector to a linear pretrained model."""
        return nonlinear_to_linear(self).apply_to(
            pretrained_linear_checkpoint, scaling_coef
        )

    def _cast_to_same_type(self, other: _TaskVector) -> Self:
        return linear_to_nonlinear(other, self.vector.keys())

    def apply_to(self, pretrained_checkpoint: os.PathLike, scaling_coef: float = 1.0) -> Dict[str, torch.Tensor]:
        """Apply a task vector to a pretrained model."""
        pretrained_state_dict = self._load_checkpoint(pretrained_checkpoint)
        if isinstance(pretrained_state_dict, nn.Module):
            pretrained_state_dict = pretrained_state_dict.state_dict()

        new_state_dict = OrderedDict()
        with torch.no_grad():
            for key in self.vector.keys():
                pt_key = key.replace('image_encoder.', '')
                new_state_dict[key] = (
                    pretrained_state_dict[pt_key] + scaling_coef * self.vector[key]
                )
        return new_state_dict


class LinearizedTaskVector(_TaskVector):
    """A task vector for linearized models."""
    def __init__(
        self, 
        pretrained_checkpoint: Optional[os.PathLike] = None, 
        finetuned_checkpoint: Optional[os.PathLike] = None, 
        vector: Optional[Self] = None,
    ):
        if vector is not None:
            self.vector = vector
        else:
            super().__init__(
                pretrained_checkpoint,
                finetuned_checkpoint,
                vector,
            )
            
            self.vector = OrderedDict()
            with torch.no_grad():
                for key in self.finetuned_state_dict.keys():
                    if self.pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue

                    self.vector[key] = (
                        self.finetuned_state_dict[key] - self.pretrained_state_dict[key]
                    )

    def _load_checkpoint(self, checkpoint: os.PathLike) -> Union[nn.Module, Dict[str, torch.Tensor]]:
        """Load a checkpoint into a model."""
        return LinearizedImageEncoder.load(checkpoint)

    def apply_to_nonlinear(
        self, pretrained_nonlinear_checkpoint: os.PathLike, param_names: Sequence[str], scaling_coef: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Apply a task vector to a nonlinear pretrained model."""
        return linear_to_nonlinear(self, param_names).apply_to(
            pretrained_nonlinear_checkpoint, scaling_coef
        )

    def apply_to_linear(self, pretrained_linear_checkpoint: os.PathLike, scaling_coef: float = 1.0) -> Dict[str, torch.Tensor]:
        """Apply a task vector to a linear pretrained model."""
        return self.apply_to(pretrained_linear_checkpoint, scaling_coef)

    def get_named_parameters(self, param_names: Sequence[str]) -> Dict[str, torch.Tensor]:
        """Get the named parameters of the task vector."""
        params = {k: v for k, v in self.vector.items() if "model.params0" not in k}
        return {k: v for k, v in zip(param_names, params.values())}

    def _cast_to_same_type(self, other: _TaskVector) -> Self:
        return nonlinear_to_linear(other)


def nonlinear_to_linear(nonlinear_task_vector: _TaskVector) -> LinearizedTaskVector:
    """Convert a nonlinear task vector to a linear task vector."""
    if isinstance(nonlinear_task_vector, LinearizedTaskVector):
        return nonlinear_task_vector
    else:
        linear_params = {
            f"model.params.{i}": v
            for i, v in enumerate(nonlinear_task_vector.vector.values())
        }
        # The diff of the init params of the linearized moodels are all zero.
        for i, v in enumerate(nonlinear_task_vector.vector.values()):
            linear_params[f"model.params0.{i}"] = torch.zeros_like(v)
        return LinearizedTaskVector(vector=linear_params)


def linear_to_nonlinear(linear_task_vector: _TaskVector, param_names: str) -> NonLinearTaskVector:
    """Convert a linear task vector to a nonlinear task vector."""
    if isinstance(linear_task_vector, NonLinearTaskVector):
        return linear_task_vector
    else:
        return NonLinearTaskVector(
            vector=linear_task_vector.get_named_parameters(param_names)
        )
