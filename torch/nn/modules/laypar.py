from collections import OrderedDict
import torch
from torch.nn.parameter import Parameter
import copy
from .module import RESOURCE_CPU, RESOURCE_GPU, Module
import torch.utils.hooks as hooks
import time

milliseconds_per_second = 1000

def find_wrapper_by_idx(wrapper_list, idx):
    for wrapper in wrapper_list:
        if wrapper._layer_idx == idx:
            return wrapper
    return None

def generate_wrapper(mod):
    wrapper_list = []

    mod = mod.cuda()
    mod.count_layers()

    for name, module in mod.named_modules():
        if module._get_name() in mod.target_layers:
            tmp = copy.deepcopy(module)
            wrapper = torch.nn.modules.Wrapper(tmp._get_name())
            wrapper._layer_idx = tmp._layer_idx
            wrapper.set_gpu_module(tmp)
            wrapper_list.append(wrapper)

    mod = mod.cpu()

    for name, module in mod.named_modules():
        if module._get_name() in mod.target_layers:
            tmp = copy.deepcopy(module)
            wrapper = find_wrapper_by_idx(wrapper_list, tmp._layer_idx)
            wrapper.set_cpu_module(tmp)

    return wrapper_list

def swap_modules(mod, wrapper_list):
    for name, module in mod.named_modules():
        for child_name, child_module in module.named_children():
            if child_name == 'cpu_module' or child_name == 'gpu_module':
                continue
            if child_module._get_name() in mod.target_layers:
                module._modules[child_name] = find_wrapper_by_idx(wrapper_list, child_module._layer_idx)


class Wrapper(Module):
    def __init__(self, name):
        super(Wrapper, self).__init__()
        self.name = name
        self.gpu_module = None
        self.cpu_module = None

    def set_gpu_module(self, mod):
        mod = copy.deepcopy(mod)
        self.gpu_module = mod

    def set_cpu_module(self, mod):
        mod = copy.deepcopy(mod)
        self.cpu_module = mod

    def _add(self, x, y):
        if self.resource == RESOURCE_CPU:
            if x.is_cuda:
                x = x.cpu()
            if y.is_cuda:
                y = y.cpu()
            return self.cpu_module.add(x, y)

        if self.resource == RESOURCE_GPU:
            if not x.is_cuda:
                x = x.cuda()
            if not y.is_cuda:
                y = y.cuda()
            return self.gpu_module.add(x, y)

    def add(self, x, y):
        print('add called!')
        result = self._add(x, y)
        return result

    def _cat(self, x, dim=0):
        new_x = tuple()

        if self.resource == RESOURCE_CPU:
            for tk in x:
                if tk.is_cuda:
                    tk = tk.cpu()
                new_x += (tk, )
            return self.cpu_module.cat(new_x, dim)

        if self.resource == RESOURCE_GPU:
            for tk in x:
                if not tk.is_cuda:
                    tk = tk.cuda()
                new_x += (tk, )
            return self.gpu_module.cat(new_x, dim)

    def cat(self, x, dim=0):
        print('cat called!')
        result = self._cat(x, dim)
        return result

    def forward(self, input):
        # print(self)
        resource = self.resource

        if not input.is_cuda and resource == RESOURCE_GPU:
            input = input.cuda()
        if input.is_cuda and resource == RESOURCE_CPU:
            input = input.cpu()

        if resource == RESOURCE_CPU:
            out = self.cpu_module(input)
            return out
        elif resource == RESOURCE_GPU:
            out = self.gpu_module(input)
            return out
        else:
            assert(f'Resource: {resource}. Wrong resource')