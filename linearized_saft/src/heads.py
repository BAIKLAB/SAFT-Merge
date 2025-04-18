import argparse
import math
import os

from tqdm import tqdm

import torch
from torch import nn
import open_clip

from tv_datasets.templates import get_templates
from tv_datasets.registry import get_dataset
from modeling import ClassificationHead, ImageEncoder


def build_classification_head(model: nn.Module, dataset_name: str, data_location: os.PathLike, device: str) -> nn.Module:
    template = get_templates(dataset_name)

    dataset = get_dataset(
        dataset_name,
        None,
        location=data_location
    )
    classnames = dataset.classnames

    logit_scale = model.logit_scale

    model.eval()
    model.to(device)

    print('Building classification head.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = open_clip.tokenize(texts).to(device) # tokenize
            embeddings = model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


def get_classification_head(args: argparse.Namespace, dataset: str) -> nn.Module:
    model = ImageEncoder(args, keep_lang=True).model
    classification_head = build_classification_head(model, dataset, args.data_location, args.device)
    return classification_head


def get_original_classification_head(args) -> nn.Module:
    filename = os.path.join(args.model_ckpt_dir, args.model, f'head_{args.dataset_name}.pt')
    if not os.path.exists(filename):
        classification_head = get_classification_head(args, args.dataset_name)
        classification_head.save(filename)
    print(f'Classification head for {args.model} on {args.dataset_name} exists at {filename}')
    return ClassificationHead.load(filename)
