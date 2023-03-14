import math
import torch
from typing import Optional
from torch.optim import Optimizer


class AdamW(Optimizer):
    r"""Implements AdamW algorithm.
    It has been proposed in `Decoupled Weight Decay Regularization`.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)
        foreach (bool, optional): whether foreach implementation of optimizer
            is used (default: None)
        capturable (bool, optional): whether this instance is safe to capture in a CUDA graph.
            Passing True can impair ungraphed performance, so if you don't intend to
            graph capture this instance, leave it False (default: False)

    ''Decoupled Weight Decay Regularization'':
        https://openreview.net/pdf?id=Bkg6RiCqY7
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, *, maximize: bool = False,
                 foreach: Optional[bool] = None,
                 capturable: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        foreach=foreach, maximize=maximize, capturable=capturable)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device) \
                        if self.defaults['capturable'] else torch.tensor(0.)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                state_steps.append(state['step'])

            for i, param in enumerate(params_with_grad):
                grad = grads[i] if not group['maximize'] else -grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step_t = state_steps[i]

                if group['capturable']:
                    assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."

                if torch.is_complex(param):
                    grad = torch.view_as_real(grad)
                    exp_avg = torch.view_as_real(exp_avg)
                    exp_avg_sq = torch.view_as_real(exp_avg_sq)
                    param = torch.view_as_real(param)

                    # update step
                    step_t += 1

                    # Perform stepweight decay
                    param.mul_(1 - group['lr'] * group['weight_decay'])

                    # Decay the first and second moment running average coefficient
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    if group['capturable']:
                        step = step_t

                        # 1 - beta1 ** step can't be captured in a CUDA graph, even if step is a CUDA tensor
                        # (incurs "RuntimeError: CUDA error: operation not permitted when stream is capturing")
                        bias_correction1 = 1 - torch.pow(beta1, step)
                        bias_correction2 = 1 - torch.pow(beta2, step)

                        step_size = group['lr'] / bias_correction1
                        step_size_neg = step_size.neg()

                        bias_correction2_sqrt = bias_correction2.sqrt()

                        if amsgrad:
                            # Maintains the maximum of all 2nd moment running avg. till now
                            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                            # Uses the max. for normalizing running avg. of gradient
                            # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                            # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                            denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(
                                group['eps'] / step_size_neg)
                        else:
                            denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(
                                group['eps'] / step_size_neg)

                        param.addcdiv_(exp_avg, denom)
                    else:
                        step = step_t.item()

                        bias_correction1 = 1 - beta1 ** step
                        bias_correction2 = 1 - beta2 ** step

                        step_size = group['lr'] / bias_correction1

                        bias_correction2_sqrt = math.sqrt(bias_correction2)

                        if amsgrad:
                            # Maintains the maximum of all 2nd moment running avg. till now
                            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                            # Use the max. for normalizing running avg. of gradient
                            denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(group['eps'])
                        else:
                            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])

                        param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
