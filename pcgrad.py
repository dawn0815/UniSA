import copy
import random
import numpy as np
import torch

class PCGradAMP():
    def __init__(self, num_tasks, optimizer: torch.optim.Optimizer, scaler: torch.cuda.amp.GradScaler = None, reduction='sum', cpu_offload: bool = False):
        self.num_tasks = num_tasks
        self.cpu_offload = cpu_offload
        self._scaler, self._optim, self._reduction = scaler, optimizer, reduction
        # Setup default accumulated gradient
        self.accum_grad = []
        for i in range(self.num_tasks):
            grad, shape, has_grad = self._retrieve_grad()
            self.accum_grad.append((grad, shape, has_grad))
        return

    def state_dict(self) -> dict:
        if self._scaler is not None:
            return {'scaler': self._scaler.state_dict(), 'optimizer': self._optim.state_dict()}
        else:
            return {'optimizer': self._optim.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        if self._scaler is not None:
            self._scaler.load_state_dict(state_dict['scaler'])
            self._optim.load_state_dict(state_dict['optimizer'])
        else:
            self._optim.load_state_dict(state_dict['optimizer'])

    @property
    def optimizer(self):
        return self._optim

    @property
    def scaler(self):
        return self._scaler

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        ret = self._optim.zero_grad()
        # Setup zero accumulated gradient
        for i in range(self.num_tasks):
            self.accum_grad[i][0].zero_()
            self.accum_grad[i][2].zero_()
        return ret

    def step(self):
        '''
        update the parameters with the gradient
        '''
        grads, shapes, has_grads = self._pack_accum_grads()
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)

        if self._scaler is not None:
            self._scaler.step(self._optim)
            self._scaler.update()
        else:
            self._optim.step()

        return self.zero_grad()

    def backward(self, mt_losses):
        # Gradient accumulation
        for loss_id, loss in enumerate(mt_losses):
            self._optim.zero_grad()
            retain_graph = (loss_id < (self.num_tasks - 1))
            if self._scaler is not None:
                self._scaler.scale(loss).backward(retain_graph = retain_graph)
            else:
                loss.backward(retain_graph=retain_graph)
            grad, shape, has_grad = self._retrieve_grad()
            acc_grad, acc_shape, acc_has_grad = self.accum_grad[loss_id]
            acc_grad += grad
            acc_has_grad = torch.logical_or(acc_has_grad, grad).to(dtype=acc_has_grad.dtype)
            self.accum_grad[loss_id] = (acc_grad, acc_shape, acc_has_grad)
        self._optim.zero_grad()

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction == 'mean':
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).sum(dim=0)
        else:
            exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx].to(p.device)
                idx += 1
        return

    def _pack_accum_grads(self):
        '''
        pack the gradient of the parameters of the network for each objective
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for (grad, shape, has_grad) in self.accum_grad:
            grads.append(grad)
            has_grads.append(has_grad)
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific
        objective
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    if self.cpu_offload:
                        grad.append(torch.zeros_like(p).cpu())
                        has_grad.append(torch.zeros_like(p, dtype=torch.int8).cpu())
                    else:
                        grad.append(torch.zeros_like(p).to(p.device))
                        has_grad.append(torch.zeros_like(p, dtype=torch.int8).to(p.device))
                else:
                    shape.append(p.grad.shape)
                    if self.cpu_offload:
                        grad.append(p.grad.detach().cpu())
                        has_grad.append(torch.ones_like(p, dtype=torch.int8).cpu())
                    else:
                        grad.append(p.grad.clone())
                        has_grad.append(torch.ones_like(p, dtype=torch.int8).to(p.device))
        grad_flatten = self._flatten_grad(grad, shape)
        has_grad_flatten = self._flatten_grad(has_grad, shape)
        return grad_flatten, shape, has_grad_flatten