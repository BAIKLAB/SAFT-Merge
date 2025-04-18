## [Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning](https://openreview.net/forum?id=eaTqsptDPL)

> [Yeoreum Lee](https://github.com/leeyeoreum02)<sup>1</sup>, Jinwook Jung<sup>1</sup>, [Sungyong Baik](https://dsybaik-hy.github.io)<sup>1,2&dagger;</sup>, <br>
> <sup> &dagger; corresponding authors </sup> <br>
> <sup>1</sup>Dept. of Artificial Intelligence, Hanyang University, <sup>2</sup>Dept. of Data Science, Hanyang University


[![Paper](https://img.shields.io/badge/Paper-ICLR_2025-blue)](https://openreview.net/forum?id=eaTqsptDPL)


Full code will be released very soon!



### Abstract
Large-scale deep learning models with a pretraining-finetuning paradigm have led to a surge of numerous task-specific models finetuned from a common pretrained model. Recently, several research efforts have been made on merging these large models into a single multi-task model, particularly with simple arithmetic on parameters. Such merging methodology faces a central challenge: interference between model parameters finetuned on different tasks. Few recent works have focused on desiging a new finetuning scheme that can lead to small parameter interference, however at the cost of the performance of each task-specific finetuned model and thereby limiting that of a merged model. To improve the performance of a merged model, we note that a finetuning scheme should aim for (1) smaller parameter interference and (2) better performance of each finetuned model on the corresponding task. In this work, we aim to design a new finetuning objective function to work towards these two goals. In the course of this process, we find such objective function to be strikingly similar to sharpness-aware minimization (SAM) objective function, which aims to achieve generalization by finding flat minima. Drawing upon our observation, we propose to finetune pretrained models via SAM or its variants. The experimental and theoretical results showcase the effectiveness and orthogonality of our proposed approach, improving performance upon various merging and finetuning methods.


### Updates
* (2025/04/18): Our code is now available!
* (2025/01/22): Our paper has been accepted at [ICLR 2025](https://iclr.cc/)🎉


## Acknowledgement
Our implementation references the code below, thanks to them.

- Task Arithmetic: https://github.com/mlfoundations/task_vectors

- Task Arithmetic in the tangent space (FTTS): https://github.com/gortizji/tangent_task_arithmetic

- Fine-Tuning Linear Layers Only Is a Simple yet Effective Way for Task Arithmetic (FTLO): https://github.com/kyrie-23/linear_task_arithmetic

- DARE: https://github.com/yule-BUAA/MergeLM

- Sharpness-Aware Minimization (SAM): https://github.com/davda54/sam


## BibTex
```
@inproceedings{
    lee2025mitigating,
    title={Mitigating Parameter Interference in Model Merging via Sharpness-Aware Fine-Tuning},
    author={Yeoreum Lee and Jinwook Jung and Sungyong Baik},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=eaTqsptDPL}
}
```
