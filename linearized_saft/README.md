## Datasets
Please follow the instructions in [this issue](https://github.com/mlfoundations/task_vectors/issues/1) to download the datasets (e.g., Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN).


## Checkpoints
Please download the checkpoints of CLIP ViT-B/{16, 32} models at [here](https://drive.google.com/file/d/1cYrCDzkDSZ39tN7Fr8S9mB3jG3L4A5Z1/view?usp=drive_link).
```sh
/path/to/checkpoints
├── model_1 (e.g., ViT-B-32)
│   ├── dataset_1 (e.g., Cars)
│   │   ├── <your own checkpoint file name 1>.pt
│   │   ├── <your own checkpoint file name 2>.pt   
│   │   ...
│   │   └── <your own checkpoint file name N>.pt
│   ├── dataset_2
│   ...
│   ├── dataset_T
│   ├── linear_zeroshot.pt
│   └── zeroshot.pt
├── model_2
...
└── model_M
```


## Dependencies
### 0. Install Poetry (w/ Python 3.8)
Please follow the instructions in https://python-poetry.org/docs/.

You can also install Poetry within a Conda environment.
### 1. Install dependencies with Poetry
```sh
poetry install
```

## Training
### 1. Linear + SGD fine-tuning
```sh
poetry run python src/finetune.py \
    --data-location /path/to/datasets \
    --train-dataset Cars,DTD,... \
    --model ViT-B-32 \
    --ft-type ftts | ftlo \
    --model-ckpt-dir /path/to/checkpoints
```
### 2. Linear + SAFT (Ours)
```sh
poetry run python src/finetune.py \
    --data-location /path/to/datasets \
    --train-dataset Cars,DTD,... \
    --model ViT-B-32 \
    --ft-type ftts | ftlo \
    --model-ckpt-dir /path/to/checkpoints \
    --is-asam \
    --rho 0.5  # default rho for ASAM
```

## Evaluation of fine-tuned models
```sh
poetry run python src/eval.py \
    --model ViT-B-32 \
    --ft-type ftts | ftlo \
    --data-location /path/to/datasets \
    --model-ckpt-dir /path/to/checkpoints \
    --ckpt-name <your own checkpoint file name>.pt \
    --eval-datasets Cars,DTD,...
```

## Merging fine-tuned models
```sh
poetry run python src/eval.py \
    --model ViT-B-32 \
    --ft-type ftts | ftlo \
    --merging-method-name average_merging | task_arithmetic | ties_merging \
    --data-location /path/to/datasets \
    --model-ckpt-dir /path/to/checkpoints \
    --ckpt-name <your own checkpoint file name>.pt \
    --source-datasets Cars,DTD,... \  # tasks to be merged
    --target-datasets Cars,DTD,...  # tasks to be evaluated
```
