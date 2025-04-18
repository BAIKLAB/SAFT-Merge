import argparse
import os
from typing import Optional

import torch
from torch import nn
import open_clip

import utils


class ImageEncoder(torch.nn.Module):
    def __init__(self, args: argparse.Namespace, keep_lang: bool = False) -> None:
        super().__init__()

        print(f'Loading {args.model} pre-trained weights.')
        if '__pretrained__' in args.model:
            name, pretrained = args.model.split('__pretrained__')
        else:
            name = args.model
            pretrained = 'openai'
        self.model, self.train_preprocess, self.val_preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, cache_dir=args.openclip_cachedir)
        
        self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        return self.model.encode_image(images)
    
    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs)

    def save(self, filename: os.PathLike) -> None:
        print(f'Saving image encoder to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, model_name: str, filename: os.PathLike) -> nn.Module:
        print(f'Loading image encoder from {filename}')
        state_dict = torch.load(filename)
        return cls.load(model_name, state_dict)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize: bool, weights: torch.Tensor, biases: Optional[torch.Tensor] = None) -> None:
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs)

    def save(self, filename: os.PathLike) -> None:
        print(f'Saving classification head to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename: os.PathLike) -> nn.Module:
        # print(f'Loading classification head from {filename}')
        return utils.torch_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder: nn.Module, classification_head: nn.Module) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self) -> None:
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs)

    def save(self, filename: os.PathLike) -> nn.Module:
        print(f'Saving image classifier to {filename}')
        utils.torch_save(self, filename)

    @classmethod
    def load(cls, filename: os.PathLike) -> nn.Module:
        print(f'Loading image classifier from {filename}')
        return utils.torch_load(filename)
