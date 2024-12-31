from Common.bcolors import bcolors
import math
import torch
from torch import nn
import os
from tqdm import tqdm
import torch.nn.functional as F
from Common.utils import MyDict
from Common.Result import Result, print_res, AverageMeter
from Common.bcolors import bcolors
from Common.classification_res import report_multi_classification, report_bin_classification
from os import path, listdir
from Common.dataset import conv2tensor
from Common.history import History_Base
import numpy as np
from os import path, listdir
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from enum import IntEnum
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
import matplotlib
matplotlib.use("Qt5Agg")
import itertools
import seaborn as sns
import plotly.express as px

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


def vcat(*vecs):
    vt_list = [v.clone().view(-1,1).cpu() if len(v.shape) == 1 else v.cpu() for v in vecs]
    print(torch.cat(vt_list, dim=1))


def hcat(*vecs):
    vt_list = [v.clone().view(1,-1) for v in vecs]
    print(torch.cat(vt_list, dim=0))


def mean_std(name, m, s, frm='.2f', echo=True, end='\n'):
    res = f'mean {name}:{m:{frm}} ± {s:{frm}}'
    if echo:
        print(res, end=end)
    return res



def load_res(file_path, key_res='res_val', whole_checkpoint=True):
    try:
        ch = torch.load(file_path)
        score, args = ch['best_score'], ch['args']
        res, hist = ch[key_res], ch['m_hist']
        results_val = ch['results_val']
        if whole_checkpoint:
            return score, args, res, hist, ch
        else:
            return score, args, res, hist, results_val
    except BaseException as err:
        print(f"Unexpected {err=}, {type(err)=}")


def disp_res(file_name, summary=True):
    import torch
    ch = torch.load(file_name)
    res, res_test, res_train = ch.get('res', None), ch['res_test'], ch['res_train']
    if res is None:
        res = ch['res_val']
    if res_test is not None:
        print_res(res_test, color=bcolors.HEADER, pre_text='Test Results', summary_sw=summary)
    print_res(res, color=bcolors.FAIL, pre_text='Validation Results', summary_sw=summary)
    print_res(res_train, color=bcolors.OKBLUE, pre_text='Train Results', summary_sw=summary)
    results_val = ch['results_val']
    print('validation Results')
    meters = MyDict(acc=[], f_score=[], precision=[], roc_auc=[])
    for i, r in enumerate(results_val):
        for m in meters.keys():
            meters[m].append(r[m])
        print_res(r, color=bcolors.HEADER, pre_text=f'fold no:{i}', summary_sw=True)
        print()

    df_m = pd.DataFrame(data=None, columns=list(meters.keys()))
    df_m.index = np.arange(1, len(df_m) + 1)
    pd.options.display.float_format = '{:.2f}'.format
    for i, r in enumerate(results_val):
        row = list(map(r.get, meters.keys()))
        df_m.loc[i] = row

    print(df_m, end='\n\n')

    for m in meters.keys():
        meters[m] = np.array(meters[m])
        mean_std(m, meters[m].mean(), meters[m].std(), echo=True)



def main():
    disp_res('./Res_SEED/eeg_dml__91.69__97.30__SEED.cvr')
    mean_std('Acc', 24.67835, 1.32545454, echo=True)
    pass


if __name__ == '__main__':
    main()
    print('Congratulations to you!')

