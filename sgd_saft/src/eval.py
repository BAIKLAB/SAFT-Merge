import argparse
import os
import json
from typing import Dict

from tqdm import tqdm

import torch
from torch import nn

import utils
from args import parse_arguments_for_eval
from tv_datasets.common import get_dataloader, maybe_dictionarize
from heads import get_original_classification_head
from modeling import ImageClassifier, ImageEncoder
from tv_datasets.registry import get_dataset


def eval_single_dataset(image_encoder: nn.Module, dataset_name: str, args: argparse.Namespace) -> Dict[str, float]:
    args.dataset_name = dataset_name
    classification_head = get_original_classification_head(args)
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()

    dataset = get_dataset(
        dataset_name,
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    dataloader = get_dataloader(dataset, is_train=False, args=args, image_encoder=None)
    device = args.device

    with torch.no_grad():
        top1 = 0.
        correct = 0. 
        n = 0.
        for data in tqdm(dataloader):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()
            
            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100*top1:.2f}%')
    
    return metrics


def eval_single_dataset_val(dataset, image_encoder: nn.Module, args: argparse.Namespace) -> Dict[str, float]:
    dataloader = get_dataloader(dataset, is_train=False, args=args)
    classification_head = get_original_classification_head(args)
    model = ImageClassifier(image_encoder, classification_head)
    model.eval()

    with torch.no_grad():
        top1 = 0.
        correct = 0. 
        n = 0.
        for data in tqdm(dataloader):
            data = maybe_dictionarize(data)
            x = data['images'].to(args.device)
            y = data['labels'].to(args.device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(args.device)

            correct += pred.eq(y.view_as(pred)).sum().item()
            
            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {args.dataset_name}. Accuracy: {100*top1:.2f}%')
    
    return metrics


def evaluate_and_save(args: argparse.Namespace) -> None:
    if args.eval_datasets is None:
        print('There are no datasets to evaluate')
        return

    for dataset_name in args.eval_datasets:
        save_result_dir = os.path.join(args.save_result_dir, args.model, dataset_name)
        os.makedirs(save_result_dir, exist_ok=True)
        save_result_path = os.path.join(save_result_dir, f'{args.ckpt_name}.json')

        if os.path.exists(save_result_path):
            print(f'{save_result_path} already exists')
            continue

        print('Evaluating on', dataset_name)

        load_model_path = os.path.join(args.model_ckpt_dir, args.model, dataset_name, args.ckpt_name)

        image_encoder = ImageEncoder(args)
        state_dict = torch.load(load_model_path).state_dict()
        image_encoder.load_state_dict(state_dict)

        results = eval_single_dataset(image_encoder, dataset_name, args)

        if 'top1' in results:
            print(f"{dataset_name} Top-1 accuracy: {results['top1']:.4f}")

        with open(save_result_path, 'w') as f:
            json.dump(results, f)
        print(f'Results saved to {save_result_path}.')


if __name__ == '__main__':
    args = parse_arguments_for_eval()
    evaluate_and_save(args)