import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer, required
import threading


class Fsgdm(Optimizer):
    r"""Implements FedAvg and Prox. Local Solver can have momentum.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        ratio (float): relative sample size of client
        gmf (float): global/server/slow momentum factor
        mu (float): parameter for proximal local SGD
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, model, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False, variance=0, v={}):
        params = model.parameters()

        # 当前的全局动量
        self.v = v
        # 每次更新的累加动量
        self.P = {}
        # 通过名字来检索对象
        self.param_name = {}
        for name, param in model.named_parameters():
            self.param_name[param] = name

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, variance=variance)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(Fsgdm, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Fsgdm, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                name = self.param_name[p]
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                param_state = self.state[p]

                if momentum != 0:
                    # 如果状态中没有或者没有初始动量，直接获取第一个梯度作为动量，适用于第 0 轮的第一次迭代
                    if 'momentum_buffer' not in param_state and name not in self.v:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    # 如果状态中没有但是提供了初始动量，初始动量作为第一动量，并执行一次动量更新，适用于 >0 轮的第一次迭代
                    elif 'momentum_buffer' not in param_state and name in self.v:
                        buf = param_state["momentum_buffer"] = torch.clone(self.v[name]).detach()
                        buf.mul_(momentum).add_(1 - momentum, d_p)
                    # 否则，从状态中拿去，并执行一次动量更新，适用于 > 1 次的迭代
                    else:
                        buf = param_state['momentum_buffer']
                        # buf.mul_(momentum).add_(1 - dampening, d_p)
                        buf.mul_(momentum).add_(1 - momentum, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    # 本地不做带动量的sgd
                    else:
                        d_p = d_p
                    # 更新动量
                    self.v[name] = buf
                    # 更新累加的本地动量，第一次克隆，之后累加
                    if name not in self.P:
                        self.P[name] = torch.clone(self.v[name]).detach()
                    else:
                        self.P[name].add_(self.v[name])

                # apply sgd update
                p.data.add_(-group['lr'], d_p)

        return loss
