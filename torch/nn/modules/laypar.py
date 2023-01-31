from collections import OrderedDict
import torch
from torch.nn.parameter import Parameter
import copy
from .module import RESOURCE_CPU, RESOURCE_GPU, Module
import torch.utils.hooks as hooks
import time
import statistics
import csv
import numpy as np

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
        self.cpu_copy_time =[]# bst
        self.gpu_forard_time=[]# bst
        self.gpu_copy_time =[]# bst
        self.cpu_forard_time=[]# bst
        self.input_size=0#bst
        self.count=0

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
        """
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
            """

        start_copy=0
        end_copy=0
        if not input.is_cuda :
            start_copy= time.time() 
            input_gpu = input.cuda()
            end_copy=time.time()
            #print("CPU to GPU COPY: {:.8f}".format(end_copy-start_copy))
            self.gpu_copy_time.append(end_copy-start_copy)
            start_copy= time.time() 
            input_cpu = input_gpu.cpu()
            end_copy=time.time()
            #print("GPU to CPU COPY: {:.8f}".format(end_copy-start_copy))
            self.cpu_copy_time.append(end_copy-start_copy)
        if input.is_cuda :
            start_copy= time.time() 
            input_cpu = input.cpu()
            end_copy=time.time()
            #print("GPU to CPU COPY: {:.8f}".format(end_copy-start_copy))
            self.cpu_copy_time.append(end_copy-start_copy)
            start_copy= time.time() 
            input_gpu = input_cpu.cuda()
            end_copy=time.time()
            #print("CPU to GPU COPY: {:.8f}".format(end_copy-start_copy))
            self.gpu_copy_time.append(end_copy-start_copy)
        start_forward= time.time() 
        out=self.cpu_module(input_cpu)
        end_forward = time.time()
        #print("BST CPU FORWARD: {:.8f}".format(end_forward-start_forward))
        self.cpu_forard_time.append(end_forward-start_forward)
        start_forward= time.time() 
        out=self.gpu_module(input_gpu)
        end_forward= time.time()
        #print("BST GPU FORWARD: {:.8f}".format(end_forward-start_forward))
        self.gpu_forard_time.append(end_forward-start_forward)
        self.count=self.count+1
        if self.count==100:
           # print("layer inputsize CPU_copy CPU_forward GPU_copy GPU_forward")
            for temp_i in range(100):
                temp_cpu_copy_time=self.cpu_copy_time[temp_i]
                temp_gpu_copy_time=self.gpu_copy_time[temp_i]
                temp_cpu_forward_time=self.cpu_forard_time[temp_i]
                temp_gpu_forward_time=self.gpu_forard_time[temp_i]
                #print(self.name, ";", self.gpu_module, ";",self.input_size, ";"," {:.8f}".format(statistics.mean(self.cpu_copy_time[1:])*1000),";"," {:.8f}".format(statistics.mean(self.cpu_forard_time[1:])*1000), ";"," {:.8f}".format(statistics.mean(self.gpu_copy_time[1:])*1000), ";"," {:.8f}".format(statistics.mean(self.gpu_forard_time[1:])*1000), ";",input.shape)
                #print("BST layer data ;", self.name, ";", self.gpu_module, ";",self.input_size, ";"," {:.8f}".format(temp_cpu_copy_time*1000),";"," {:.8f}".format(temp_cpu_forward_time*1000), ";"," {:.8f}".format(temp_gpu_copy_time*1000), ";"," {:.8f}".format(temp_gpu_forward_time*1000), ";",input.shape)
            #for copy_temp in self.cpu_copy_time:
            #    print("cpu_copy_time", self.input_size, " {:.8f}".format(copy_temp))
            with open('/media/bst/hdd/mirae/layer-par/layer-pytorch/test/default_baseline_taskset.csv', 'a') as f_object:
                writer = csv.writer(f_object)
                array_0 = np.array([self.name, "BST CPU Average copy", round(statistics.mean(self.cpu_copy_time[1:])*1000, 8)])
                array_1 = np.array([self.name, "BST CPU Average FORWARD", round(statistics.mean(self.cpu_forard_time[1:])*1000, 8)])
                array_2 = np.array([self.name, "BST GPU Average copy", round(statistics.mean(self.gpu_copy_time[1:])*1000, 8)])
                array_3 = np.array([self.name, "BST GPU Average FORWARD", round(statistics.mean(self.gpu_forard_time[1:])*1000, 8)]) 
                writer.writerow(array_0)
                writer.writerow(array_1)
                writer.writerow(array_2)
                writer.writerow(array_3)
            
            print(self.name, "BST CPU Average copy: {:.8f}".format(statistics.mean(self.cpu_copy_time[1:])*1000))
            print(self.name, "BST CPU Average FORWARD: {:.8f}".format(statistics.mean(self.cpu_forard_time[1:])*1000))
            print(self.name, "BST GPU Average copy: {:.8f}".format(statistics.mean(self.gpu_copy_time[1:])*1000))
            print(self.name, "BST GPU Average FORWARD: {:.8f}".format(statistics.mean(self.gpu_forard_time[1:])*1000))
        return out

