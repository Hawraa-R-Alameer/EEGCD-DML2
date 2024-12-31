from collections import defaultdict
import textwrap
import random
import numpy as np
import torch
from .bcolors import bcolors


class MyDict(dict):
    def __init__(self, *args, **kwargs):
        super(MyDict, self).__init__(*args, **kwargs)
        for k in self.keys():
            v = self[k]
            if type(v) == dict:
                self[k] = MyDict(v)

    def __getattr__(self, name):
        # return self[name]
        try:
            return self[name]
        except KeyError:
            if 'default_val' in self.keys():
                return self['default_val']
            raise AttributeError(name)

    def get(self, key, default_val=None):
        try:
            return self[key]
        except:
            return default_val

    def __setattr__(self, key, value):
        if type(value) == dict:
            self[key] = MyDict(value)
        else:
            self[key] = value

    # check boolean key
    def check_key(self, key):
        return key in self and self[key]

    def print(self, name='params', width=150):
        print(name)
        print(textwrap.fill(str(self), width=width))

    def sprint(self, name='params', width=150, sep='\n'):
        return name + sep + textwrap.fill(str(self), width=width)


class Flags_Base(MyDict):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _check_eq(flag, val):
        return True if flag == 'all' or (flag == val) else False

    def match(self, args):
        for fl in self.keys():
            if fl in args and (not Flags_Base._check_eq(self[fl], args[fl])):
                return False
        return True


class AttrDict(defaultdict):
    def __init__(self):
        super(AttrDict, self).__init__(AttrDict)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def cls_distribution(y):
    if torch.is_tensor(y):
        return torch.round(100 * y.bincount() / len(y), decimals=2)
    return np.round(100 * np.bincount(y) / len(y), 2)


def mean_std_str(m, s, format_m='.2f', format_s='.2f'):
    return f'{m:{format_m}}±{s:{format_s}}'


def disp_mean_std(m, s, desc='', format_m='.2f', format_s='.2f', color=bcolors.FAIL, sep=':'):
    res = mean_std_str(m, s, format_m, format_s)
    print(color + f'{desc}{sep} {res}' + bcolors.ENDC)


def reset_rng(seed=0):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if (torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)  # set random seed for all gpus
        # torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False


def seed_all(seed=0):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if (torch.cuda.is_available()):
        torch.cuda.manual_seed_all(seed)  # set random seed for all gpus
        # torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False

def conv2tensor(*l):
    return [X if torch.is_tensor(X) else torch.from_numpy(X) for X in l]


def conv2tensor2(data_list, device='cpu'):
    for i in range(len(data_list)):
        if data_list[i] is None:
            continue
        if isinstance(data_list[i], list):
            data_list[i] = torch.tensor(data_list[i]).to(device)
        else:
            data_list[i] = data_list[i].to(device) if torch.is_tensor(data_list[i]) else \
                torch.from_numpy(data_list[i]).to(device)
    return data_list


def num_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def dip_cuda_usage():
    device = 'cuda'
    free = torch.cuda.mem_get_info()[0] / (1024 ** 2)
    used = torch.cuda.memory_allocated(device) / (1024 ** 2)
    print(f'Cuda free:{free} MB, used:{used} MB')


def main():
    disp_mean_std(12.678, 2.45, 'mean of pixel vals', '.3f', color=bcolors.HEADER)
    ds_dict = MyDict({
        "mediaeval2015": {'text_only': True, 'path': './data'},
        "mediaeval_2016": True,
        'weibo': False, 'politi_fact': False, 'gossip': False, 'covid19_fake': True,
        'default_val': {'text_only': True, 'path': './data'}
    })

    ds_dict.test_db = False
    print('finished!!!')


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
