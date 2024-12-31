from common.bcolors import bcolors
import einops
import imageio
import torch
from torch import nn
from torch.optim import Adam, AdamW, RMSprop
import torch.nn.functional as F
import torchvision.transforms as transforms
from common.utils import MyDict
from common.bcolors import bcolors
import numpy as np
from os import path, listdir
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from common.Result import Result, print_res, AverageMeter
from common.classification_res import report_multi_classification, report_bin_classification
from enum import IntEnum
import matplotlib.pyplot as plt
import pandas as pd
import math
import matplotlib
matplotlib.use("Qt5Agg")
import os
from tqdm import tqdm
from common.history import History_Base
import random
import itertools
import seaborn as sns
from sklearn.manifold import TSNE
from glob import glob
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import yaml
from copy import deepcopy

class Algs(IntEnum):
    eeg_dml = 0
    eeg_cd = 5
    clf_base = 1
    clf_adv = 2
    soft_triple = 3
    clf_base_DE = 4


old_repr = torch.Tensor.__repr__
def tensor_info(tensor):
    if len(tensor.shape) > 0:
        return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n ' + old_repr(tensor)
    else:
        return old_repr(tensor) + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device)

torch.Tensor.__repr__ = tensor_info


def dip_cuda_usage():
    device = 'cuda'
    free = torch.cuda.mem_get_info()[0] / (1024 ** 2)
    used = torch.cuda.memory_allocated(device) / (1024 ** 2)
    print(f'Cuda free:{free} MB, used:{used} MB')