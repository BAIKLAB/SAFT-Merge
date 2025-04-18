import argparse
from collections import OrderedDict
import copy
import json
import os
from typing import Dict, Sequence

import torch
from torch import nn

from args import parse_arguments_for_merge
from tv_datasets.registry import get_dataset
from eval import eval_single_dataset, eval_single_dataset_val
from modeling import ImageEncoder
from task_vectors import TaskVector
from utils import set_random_seed


DATASET_NAMES = ['Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SUN397', 'SVHN']
MERGING_METHOD_NAMES = ['average_merging', 'task_arithmetic', 'ties_merging']


set_random_seed(seed=0)


def average_merging(
    model_paths: Sequence[os.PathLike], 
    device: str,
    **kwargs
) -> Dict[str, torch.Tensor]:
    averaged_state_dict = OrderedDict()
    
    with torch.no_grad():
        for model_idx in range(len(model_paths)):
            model_to_merge = torch.load(model_paths[model_idx])
            model_to_merge = model_to_merge.to(device)
            param_dict = copy.deepcopy(model_to_merge.state_dict())
            
            for param_name in param_dict.keys():
                if model_idx == 0:
                    averaged_state_dict[param_name] = param_dict[param_name]
                else:
                    averaged_state_dict[param_name] += param_dict[param_name]

        for param_name in param_dict.keys():
            averaged_state_dict[param_name] /= len(model_paths)

    return averaged_state_dict, None


def task_arithmetic(
    model_paths: Sequence[os.PathLike], 
    args: argparse.Namespace,
    merged_task_vector: TaskVector,
    scaling_coefficient_range: Sequence[float] = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    **kwargs
) -> Dict[str, torch.Tensor]:
    root_dir = os.path.dirname(os.path.dirname(model_paths[0]))
    pretrained_checkpoint = os.path.join(root_dir, 'zeroshot.pt')

    best_accuracy = 0.
    best_merged_params = None
    best_coeff = {}
    for scaling_coefficient in scaling_coefficient_range:
        with torch.no_grad():
            merged_state_dict = merged_task_vector.apply_to(
                pretrained_checkpoint=pretrained_checkpoint, 
                scaling_coef=scaling_coefficient,
            )
        image_encoder = ImageEncoder(args)
        image_encoder.load_state_dict(merged_state_dict)

        print(f'Evaluating on {args.dataset_name} with scaling_coefficient={scaling_coefficient}')
        dataset = get_dataset(
            f'{args.dataset_name}Val',
            image_encoder.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size,
        )
        results = eval_single_dataset_val(dataset, image_encoder, args)
        accuracy = results['top1']
        if accuracy > best_accuracy:
            print(f'better model is saved: {best_accuracy} < {accuracy}')
            best_accuracy = accuracy
            best_merged_params = copy.deepcopy(merged_state_dict)
            best_coeff['scaling_coefficient'] = scaling_coefficient

    return best_merged_params, best_coeff


def task_vector_param_dict_to_single_vector(task_vector: TaskVector) -> torch.Tensor:
    sorted_task_vector_param_dict = OrderedDict(sorted(task_vector.vector.items()))

    return nn.utils.parameters_to_vector([param.flatten() for param in sorted_task_vector_param_dict.values()])

def single_vector_to_task_vector_param_dict(single_vector: torch.Tensor, task_vector: TaskVector) -> Dict[str, torch.Tensor]:
    sorted_task_vector_param_dict = OrderedDict(sorted(task_vector.vector.items()))

    nn.utils.vector_to_parameters(single_vector, sorted_task_vector_param_dict.values())

    return sorted_task_vector_param_dict

def mask_smallest_magnitude_param_values(flattened_models_to_merge_param: torch.Tensor, param_value_mask_rate: float = 0.8) -> torch.Tensor:
    num_mask_params = int(flattened_models_to_merge_param.shape[1] * param_value_mask_rate)

    kth_values, _ = flattened_models_to_merge_param.abs().kthvalue(k=num_mask_params, dim=1, keepdim=True)
    mask = flattened_models_to_merge_param.abs() >= kth_values

    return flattened_models_to_merge_param * mask

def get_param_signs(flattened_models_to_merge_param: torch.Tensor) -> torch.Tensor:
    param_signs = torch.sign(flattened_models_to_merge_param.sum(dim=0))
    majority_sign = torch.sign(param_signs.sum(dim=0))
    param_signs[param_signs == 0] = majority_sign
    return param_signs

def disjoint_merge(flattened_models_to_merge_param: torch.Tensor, param_signs: torch.Tensor) -> torch.Tensor:
    param_to_preserve_mask = ((param_signs.unsqueeze(dim=0) > 0) & (flattened_models_to_merge_param > 0)) | ((param_signs.unsqueeze(dim=0) < 0) & (flattened_models_to_merge_param < 0))
    param_to_preserve = flattened_models_to_merge_param * param_to_preserve_mask

    num_models_param_preserved = (param_to_preserve != 0).sum(dim=0).float()
    merged_flattened_param = torch.sum(param_to_preserve, dim=0) / torch.clamp(num_models_param_preserved, min=1.0)

    return merged_flattened_param


def _ties_merging(models_to_merge_task_vectors, param_value_mask_rate):
    models_to_merge_task_vectors = copy.deepcopy(models_to_merge_task_vectors)

    flattened_models_to_merge_param = torch.vstack([
        task_vector_param_dict_to_single_vector(task_vector) 
        for task_vector in models_to_merge_task_vectors
    ])

    flattened_models_to_merge_param = mask_smallest_magnitude_param_values(
        flattened_models_to_merge_param=flattened_models_to_merge_param, 
        param_value_mask_rate=param_value_mask_rate,
    )

    param_signs = get_param_signs(
        flattened_models_to_merge_param=flattened_models_to_merge_param
    )

    merged_flattened_param = disjoint_merge(
        flattened_models_to_merge_param=flattened_models_to_merge_param, 
        param_signs=param_signs,
    )

    merged_task_vector_param_dict = single_vector_to_task_vector_param_dict(
        single_vector=merged_flattened_param, 
        task_vector=models_to_merge_task_vectors[0]
    )

    return merged_task_vector_param_dict


def ties_merging(
    model_paths: Sequence[os.PathLike],
    args: argparse.Namespace,
    models_to_merge_task_vectors,
    scaling_coefficient_range: Sequence[float] = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    param_value_mask_rate_range: Sequence[float] = [0.7, 0.8, 0.9],
    **kwargs
) -> Dict[str, torch.Tensor]:
    root_dir = os.path.dirname(os.path.dirname(model_paths[0]))
    pretrained_checkpoint = os.path.join(root_dir, 'zeroshot.pt')

    image_encoder = ImageEncoder(args)

    with torch.no_grad():
        best_accuracy = 0.
        best_merged_params = None
        best_coeff = {}
        for scaling_coefficient in scaling_coefficient_range:
            for param_value_mask_rate in param_value_mask_rate_range:
                assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

                merged_task_vector_param_dict = _ties_merging(
                    models_to_merge_task_vectors, 
                    param_value_mask_rate,
                )
                merged_task_vector = TaskVector(vector=merged_task_vector_param_dict, is_cuda=False)
                merged_params = merged_task_vector.apply_to(
                    pretrained_checkpoint=pretrained_checkpoint, 
                    scaling_coef=scaling_coefficient,
                )
                
                image_encoder.load_state_dict(merged_params)

                print(
                    f'Evaluating on {args.dataset_name} with scaling_coefficient={scaling_coefficient} '
                    f'and param_value_mask_rate={param_value_mask_rate}'
                )
                dataset = get_dataset(
                    f'{args.dataset_name}Val',
                    image_encoder.val_preprocess,
                    location=args.data_location,
                    batch_size=args.batch_size,
                )
                results = eval_single_dataset_val(dataset, image_encoder, args)
                accuracy = results['top1']
                if accuracy > best_accuracy:
                    print(f'better model is saved: {best_accuracy} < {accuracy}')
                    best_accuracy = accuracy
                    best_merged_params = copy.deepcopy(merged_params)
                    best_coeff['scaling_coefficient'] = scaling_coefficient
                    best_coeff['param_value_mask_rate'] = param_value_mask_rate

    return best_merged_params, best_coeff


MERGING_METHODS = {
    'average_merging': average_merging,
    'task_arithmetic': task_arithmetic,
    'ties_merging': ties_merging,
}


def merge_ckpts(args: argparse.Namespace) -> None:
    if args.merging_method_name not in MERGING_METHODS.keys():
        raise NotImplementedError

    load_model_paths = []
    for source_dataset_name in args.source_datasets:
        load_model_path = os.path.join(args.model_ckpt_dir, args.model, source_dataset_name, args.ckpt_name)
        print(f'loading a checkpoint from {load_model_path}')
        load_model_paths.append(load_model_path)

    pretrained_checkpoint = os.path.join(args.model_ckpt_dir, args.model, 'zeroshot.pt')

    models_to_merge_task_vectors = None
    merged_task_vector = None
    if args.merging_method_name in ['task_arithmetic', 'ties_merging']:
        models_to_merge_task_vectors = [
            TaskVector(
                pretrained_checkpoint=pretrained_checkpoint, 
                finetuned_checkpoint=model_path,
            ) 
            for model_path in load_model_paths
        ]
        merged_task_vector = sum(models_to_merge_task_vectors)

    for target_dataset_name in args.target_datasets:
        assert args.merging_method_name in MERGING_METHOD_NAMES, \
            f'merging_method_name must be one of {MERGING_METHOD_NAMES}'
        
        dataset_to_merge = '_'.join(args.source_datasets)
        save_result_dir = os.path.join(args.save_result_dir, args.model, f'{target_dataset_name}__{dataset_to_merge}', args.merging_method_name)
        os.makedirs(save_result_dir, exist_ok=True)
        json_fn = os.path.splitext(args.ckpt_name)[0]
        save_result_path = os.path.join(save_result_dir, f'{json_fn}.json')
        
        if os.path.exists(save_result_path):
            print(f'{save_result_path} already exists')
            continue
        
        args.dataset_name = target_dataset_name
        merged_state_dict, best_coeff = MERGING_METHODS[args.merging_method_name](
            load_model_paths, 
            merged_task_vector=merged_task_vector,
            models_to_merge_task_vectors=models_to_merge_task_vectors,
            device=args.device, 
            args=args,
        )
        image_encoder = ImageEncoder(args)
        image_encoder.load_state_dict(merged_state_dict)

        print('Evaluating on', target_dataset_name)
        results = eval_single_dataset(image_encoder, target_dataset_name, args)
        if 'top1' in results:
            print(f"{target_dataset_name} Top-1 accuracy: {results['top1']:.4f}")

        if best_coeff is not None:
            for k, v in best_coeff.items():
                results[k] = v

        with open(save_result_path, 'w') as f:
            json.dump(results, f)
        print(f'Results saved to {save_result_path}.')


if __name__ == '__main__':
    args = parse_arguments_for_merge()
    merge_ckpts(args)
