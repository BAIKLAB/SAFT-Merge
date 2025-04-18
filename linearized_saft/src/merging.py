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
from utils import set_random_seed, DotDict
from linear_task_vectors import LinearizedTaskVector, NonLinearTaskVector, _TaskVector, linear_to_nonlinear
from linearize import LinearizedImageEncoder, ReluEncoder


DATASET_NAMES = ['Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SUN397', 'SVHN']
MERGING_METHOD_NAMES = ['mask_merging', 'average_merging', 'task_arithmetic', 'fisher_merging', 'regmean_merging', 'ties_merging']


set_random_seed(seed=0)


def load_state_dict(state_dict: Dict[str, torch.Tensor], args: argparse.Namespace) -> nn.Module:
    if args.ft_type == 'ftts':
        args_for_load = DotDict({'model': args.model})
        args_for_load.model = args_for_load.model.split('_')[0]
        image_encoder = LinearizedImageEncoder(args_for_load)
    else:
        image_encoder = ImageEncoder(args, keep_lang=False)
        image_encoder = ReluEncoder(args, image_encoder=image_encoder)
    image_encoder.load_state_dict(state_dict)
    return image_encoder


def average_merging(
    model_paths: Sequence[os.PathLike], args: argparse.Namespace, **kwargs
) -> Dict[str, torch.Tensor]:
    averaged_state_dict = OrderedDict()

    with torch.no_grad():
        for model_idx, model_path in enumerate(model_paths):
            if args.ft_type == 'ftts':
                linear_param_dict = torch.load(model_path)
                linear_param_dict.pop('model_name')
                param_dict = {
                    k: v for k, v in linear_param_dict.items() if k.split('.')[1] == 'params'
                }
            else:
                param_dict = torch.load(model_path)
                param_dict.pop('model_name', None)
            
            for param_name in param_dict.keys():
                if model_idx == 0:
                    averaged_state_dict[param_name] = param_dict[param_name]
                else:
                    averaged_state_dict[param_name] += param_dict[param_name].cpu()

        # breakpoint()
        for param_name in param_dict.keys():
            averaged_state_dict[param_name] /= len(model_paths)

    if args.ft_type == 'ftts':
        for k in linear_param_dict.keys():
            if k.split('.')[1] == 'params0':
                averaged_state_dict[k] = linear_param_dict[k]

    return averaged_state_dict, None


def task_arithmetic(
    model_paths: Sequence[os.PathLike], 
    args: argparse.Namespace,
    merged_task_vector: _TaskVector,
    scaling_coefficient_range: Sequence[float] = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    **kwargs
) -> Dict[str, torch.Tensor]:
    zeroshot_ckpt_name = 'linear_zeroshot.pt' if args.ft_type == 'ftts' else 'zeroshot.pt'
    root_dir = os.path.dirname(os.path.dirname(model_paths[0]))
    pretrained_checkpoint = os.path.join(root_dir, zeroshot_ckpt_name)

    best_accuracy = 0.
    best_merged_params = None
    best_coeff = {}
    for scaling_coefficient in scaling_coefficient_range:
        with torch.no_grad():
            merged_state_dict = merged_task_vector.apply_to(
                pretrained_checkpoint=pretrained_checkpoint, scaling_coef=scaling_coefficient
            )

        image_encoder = load_state_dict(merged_state_dict, args)

        print(f'Evaluating on {args.dataset_name} with scaling_coefficient={scaling_coefficient}')
        dataset = get_dataset(
            f'{args.dataset_name}Val',
            image_encoder.val_preprocess,
            location=args.data_location,
            batch_size=args.batch_size
        )
        results = eval_single_dataset_val(dataset, image_encoder, args)
        accuracy = results['top1']
        if accuracy > best_accuracy:
            print(f'better model is saved: {best_accuracy} < {accuracy}')
            best_accuracy = accuracy
            best_merged_params = copy.deepcopy(merged_state_dict)
            best_coeff['alpha'] = scaling_coefficient

    return best_merged_params, best_coeff


def ties_merging(
    model_paths: Sequence[os.PathLike],
    args: argparse.Namespace,
    models_to_merge_task_vectors: Sequence[_TaskVector],
    scaling_coefficient_range: Sequence[float] = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    param_value_mask_rate_range: Sequence[float] = [0.7, 0.8, 0.9],
    **kwargs
) -> Dict[str, torch.Tensor]:
    zeroshot_ckpt_name = 'linear_zeroshot.pt' if args.ft_type == 'ftts' else 'zeroshot.pt'
    root_dir = os.path.dirname(os.path.dirname(model_paths[0]))
    pretrained_checkpoint = os.path.join(root_dir, zeroshot_ckpt_name)
    
    if args.ft_type == 'ftts':
        nonlinear_pt_ckpt = os.path.join(root_dir, 'zeroshot.pt')
        nonlinear_state_dict = torch.load(nonlinear_pt_ckpt).state_dict()
    else:
        image_encoder = ReluEncoder(args)
        finetuned_param_keys = []
        for k, v in image_encoder.named_parameters():
            if v.requires_grad:
                finetuned_param_keys.append(k)

    best_accuracy = 0.
    best_merged_params = None
    best_coeff = {}
    for scaling_coefficient in scaling_coefficient_range:
        for param_value_mask_rate in param_value_mask_rate_range:
            assert isinstance(scaling_coefficient, float), "wrong type of scaling_coefficient, should be float!"

            with torch.no_grad():
                flattened_params = []
                if args.ft_type == 'ftlo':
                    finetuned_flattened_params = []
                for tv in models_to_merge_task_vectors:
                    if args.ft_type == 'ftts':
                        tv = linear_to_nonlinear(tv, nonlinear_state_dict.keys())
                    flattened_tv = nn.utils.parameters_to_vector(tv.vector.values())
                    flattened_params.append(flattened_tv)
                    if args.ft_type == 'ftlo':
                        filtered_params = []
                        for k in finetuned_param_keys:
                            filtered_params.append(tv.vector[k])
                        finetuned_flattened_tv = nn.utils.parameters_to_vector(filtered_params)
                        finetuned_flattened_params.append(finetuned_flattened_tv)

                flattened_params = torch.vstack(flattened_params)
                if args.ft_type == 'ftlo':
                    finetuned_flattened_params = torch.vstack(finetuned_flattened_params)

                abs_flattened_params = flattened_params.abs()
                if args.ft_type == 'ftts':
                    num_mask_params = int(abs_flattened_params.shape[1] * param_value_mask_rate)
                    kth_values = abs_flattened_params.kthvalue(k=num_mask_params, dim=1, keepdim=True)[0]
                else:
                    abs_finetuned_flattened_params = finetuned_flattened_params.abs()
                    num_mask_params = int(abs_finetuned_flattened_params.shape[1] * param_value_mask_rate)
                    kth_values = abs_finetuned_flattened_params.kthvalue(k=num_mask_params, dim=1, keepdim=True)[0]

                mask = abs_flattened_params >= kth_values
                flattened_params = flattened_params * mask

                final_signs = flattened_params.sum(dim=0).sign()

                disjoint_mask = flattened_params.sign() == final_signs
                num_filtered_models = disjoint_mask.sum(dim=0)

                final_flattened_params = flattened_params * disjoint_mask
                final_flattened_params = final_flattened_params.sum(dim=0) / num_filtered_models

                merged_param_dict = OrderedDict()
                param_idx = 0
                if args.ft_type == 'ftts':
                    for k, v in nonlinear_state_dict.items():
                        num_params = v.numel()
                        merged_param_dict[k] = final_flattened_params[param_idx:param_idx+num_params].reshape_as(v)
                        param_idx += num_params
                else:
                    for k, v in models_to_merge_task_vectors[0].vector.items():
                        num_params = v.numel()
                        merged_param_dict[k] = final_flattened_params[param_idx:param_idx+num_params].reshape_as(v)
                        param_idx += num_params
                
                merged_task_vector = NonLinearTaskVector(vector=merged_param_dict)
                if args.ft_type == 'ftts':
                    merged_params = merged_task_vector.apply_to_linear(
                        pretrained_linear_checkpoint=pretrained_checkpoint,
                        scaling_coef=scaling_coefficient,
                    )
                else:
                    merged_params = merged_task_vector.apply_to(
                        pretrained_checkpoint=pretrained_checkpoint, 
                        scaling_coef=scaling_coefficient,
                    )
            
            image_encoder = load_state_dict(merged_params, args)

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

    zeroshot_ckpt_name = 'linear_zeroshot.pt' if args.ft_type == 'ftts' else 'zeroshot.pt'
    root_dir = os.path.dirname(os.path.dirname(load_model_paths[0]))
    pretrained_checkpoint = os.path.join(root_dir, zeroshot_ckpt_name)

    merged_task_vector = None
    models_to_merge_task_vectors = None
    if args.merging_method_name in ['task_arithmetic', 'ties_merging']:
        task_vector_type = LinearizedTaskVector if args.ft_type == 'ftts' else NonLinearTaskVector
        models_to_merge_task_vectors = [
            task_vector_type(
                pretrained_checkpoint, 
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

        image_encoder = load_state_dict(merged_state_dict, args)

        print('Evaluating on', target_dataset_name)
        results = eval_single_dataset(image_encoder, target_dataset_name, args)
        
        if best_coeff is not None:
            for k, v in best_coeff.items():
                results[k] = v
            print(f'results: {results}')

        with open(save_result_path, 'w') as f:
            json.dump(results, f)
        print(f'Results saved to {save_result_path}.')


if __name__ == '__main__':
    args = parse_arguments_for_merge()
    merge_ckpts(args)
