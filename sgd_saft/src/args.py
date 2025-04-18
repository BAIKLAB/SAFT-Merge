import os
import argparse

import torch


def parse_arguments_for_train() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/datasets'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--model-ckpt-dir",
        type=str,
        default=os.path.expanduser('~/checkpoints'),
        help="The root directory for the encoder checkpoint.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default=os.path.expanduser('~/.cache/open_clip'),
        help='Directory for caching models from OpenCLIP'
    )
    parser.add_argument(
        "--is-sam",
        action='store_true',
        default=False,
        help='merge with SAM fine-tuned checkpoints'
    )
    parser.add_argument(
        "--is-asam",
        action='store_true',
        default=False,
        help='merge with ASAM fine-tuned checkpoints'
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.05,
    )
    parsed_args = parser.parse_args()
    parsed_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return parsed_args


def parse_arguments_for_merge() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--merging-method-name",
        type=str,
        help="Name of merging method",
    )
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/datasets'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-ckpt-dir",
        type=str,
        default=os.path.expanduser('~/checkpoints'),
        help="The root directory for the encoder checkpoint.",
    )
    parser.add_argument(
        "--ckpt-name",
        type=str,
        help="The file name of the encoder checkpoint.",
    )
    parser.add_argument(
        "--save-result-dir",
        type=str,
        default='./results',
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--source-datasets",
        type=lambda x: x.split(","),
        help="Which datasets to use for merging. Split by comma, e.g. MNIST,EuroSAT. "
    )
    parser.add_argument(
        "--target-datasets",
        type=lambda x: x.split(","),
        help="Which datasets to use for merging and evaluation. Split by comma, e.g. MNIST,EuroSAT. "
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default=os.path.expanduser('~/.cache/open_clip'),
        help='Directory for caching models from OpenCLIP'
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return parsed_args


def parse_arguments_for_eval() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/datasets'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-ckpt-dir",
        type=str,
        default=os.path.expanduser('~/checkpoints'),
        help="The root directory for the encoder checkpoint.",
    )
    parser.add_argument(
        "--ckpt-name",
        type=str,
        help="The file name of the encoder checkpoint.",
    )
    parser.add_argument(
        "--save-result-dir",
        type=str,
        default='./results',
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--eval-datasets",
        type=lambda x: x.split(","),
        help="Which datasets to use for merging. Split by comma, e.g. MNIST,EuroSAT. "
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default=os.path.expanduser('~/.cache/open_clip'),
        help='Directory for caching models from OpenCLIP'
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return parsed_args

