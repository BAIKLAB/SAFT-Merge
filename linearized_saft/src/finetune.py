import argparse
import os
import time

import torch

from args import parse_arguments_for_train
from linearize import LinearizedImageEncoder, ReluEncoder
from tv_datasets.common import get_dataloader, maybe_dictionarize
from tv_datasets.registry import get_dataset
from eval import eval_single_dataset_val
from modeling import ImageEncoder, ImageClassifier
from utils import cosine_lr, LabelSmoothing, use_sam_variants
from heads import get_original_classification_head
from sam import SAM


def finetune(args: argparse.Namespace) -> None:
    assert args.dataset_name is not None, 'Please provide a training dataset.'

    ckpdir = os.path.join(args.save, args.dataset_name)

    ft_prefix = 'linear' if args.ft_type == 'ftts' else 'relu'

    # Check if checkpoints already exist
    ckpt_name = f'{ft_prefix}_finetuned'
    if args.is_sam:
        ckpt_name = f'{ckpt_name}_sam' if args.rho == 0.05 else f'{ckpt_name}_sam_rho{args.rho}'
    elif args.is_asam:
        ckpt_name = f'{ckpt_name}_asam' if args.rho == 0.5 else f'{ckpt_name}_asam_rho{args.rho}'

    ft_last_path = os.path.join(ckpdir, f'{ckpt_name}_v2_4x_last.pt')
    ft_best_path = os.path.join(ckpdir, f'{ckpt_name}_v2_4x_best.pt')
    if os.path.exists(ft_last_path) and os.path.exists(ft_best_path):
        print(f'Skipping fine-tuning because finetuned checkpoints exist.')
        return

    print('Building image encoder.')
    if args.ft_type == 'ftts':
        image_encoder = LinearizedImageEncoder(args, keep_lang=False)
    else:
        image_encoder = ImageEncoder(args, keep_lang=False)
        image_encoder = ReluEncoder(args, image_encoder=image_encoder)

    classification_head = get_original_classification_head(args)

    model = ImageClassifier(image_encoder, classification_head)
    model.freeze_head()
    model = model.cuda()

    preprocess_fn = model.train_preprocess
    print_every = 100

    dataset = get_dataset(
        f'{args.dataset_name}Val',
        preprocess_fn,
        location=args.data_location,
        batch_size=args.batch_size
    )
    num_batches = len(dataset.train_loader)
    data_loader = get_dataloader(dataset, is_train=True, args=args, image_encoder=None)
    
    if args.ls > 0:
        loss_fn = LabelSmoothing(args.ls)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    params = [p for p in model.parameters() if p.requires_grad]
    if use_sam_variants(args):
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(
            params,
            base_optimizer, 
            lr=args.lr,
            weight_decay=args.wd,
            rho=args.rho, 
            adaptive=args.is_asam,
        )
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    if use_sam_variants(args):
        scheduler = cosine_lr(
            optimizer.base_optimizer, 
            args.lr, 
            args.warmup_length, 
            args.epochs * num_batches // args.num_grad_accumulation
        )
    else:
        scheduler = cosine_lr(
            optimizer, 
            args.lr, 
            args.warmup_length, 
            args.epochs * num_batches // args.num_grad_accumulation
        )

    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)

    best_acc = 0.
    for epoch in range(args.epochs):
        model.train()
        for i, batch in enumerate(data_loader):
            def closure():
                loss = loss_fn(model(inputs), labels)
                loss.backward()
                return loss
            
            start_time = time.time()

            step = (
                i // args.num_grad_accumulation 
                + epoch * num_batches // args.num_grad_accumulation
            )

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to('cuda:0')
            labels = batch['labels'].to('cuda:0')
            data_time = time.time() - start_time

            loss = loss_fn(model(inputs), labels)

            loss.backward()

            if (i + 1) % args.num_grad_accumulation == 0:
                scheduler(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step(closure) if use_sam_variants(args) else optimizer.step()

            batch_time = time.time() - start_time

            if (
                step % print_every == 0
                and ((i + 1) % args.num_grad_accumulation == 0)
            ):
                percent_complete = 100 * i / len(data_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

        # Evaluate
        if args.save is not None:
            image_encoder = model.image_encoder
            results = eval_single_dataset_val(dataset, image_encoder, args)
            acc = results['top1']
            if best_acc < acc:
                best_acc = acc
                image_encoder.save(ft_best_path)
                print(f'best checkpoint is saved at {ft_best_path}')

    if args.save is not None:
        image_encoder.save(ft_last_path)
        print(f'last checkpoint is saved at {ft_last_path}')
        return


if __name__ == '__main__':
    epochs = {
        'Cars': 140,
        'DTD': 304,
        'EuroSAT': 48,
        'GTSRB': 44,
        'MNIST': 20,
        'RESISC45': 60,
        'SUN397': 56,
        'SVHN': 16,
    }
    batch_sizes = {
        'ViT-B-32': 128,
        'ViT-B-16': 128,
        'ViT-L-14': 32,
    }

    args = parse_arguments_for_train()
    args.num_grad_accumulation = 4 if args.model == 'ViT-L-14' else 1

    for dataset in args.train_dataset:
        print('='*100)
        print(f'Finetuning {args.model} on {dataset}')
        print('='*100)
        
        args.epochs = epochs[dataset]
        args.dataset_name = dataset
        args.batch_size = batch_sizes[args.model]
        args.save = os.path.join(args.model_ckpt_dir, args.model)
        finetune(args)
