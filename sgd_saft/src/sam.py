from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, Optional

import torch


class BaseSAM(torch.optim.Optimizer, metaclass=ABCMeta):
    def __init__(
        self, 
        params, 
        base_optimizer: torch.optim.Optimizer, 
        rho: float, 
        adaptive: bool = False,
        **kwargs
    ) -> None:
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @abstractmethod
    def first_step(self, zero_grad: bool = False) -> None:
        raise NotImplementedError

    @abstractmethod
    def second_step(self, zero_grad: bool = False) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> None:
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class SAM(BaseSAM):
    def __init__(
        self, 
        params, 
        base_optimizer: torch.optim.Optimizer, 
        rho: float,
        adaptive: bool = False,
        **kwargs
    ) -> None:
        super().__init__(params, base_optimizer, rho, adaptive, **kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()
