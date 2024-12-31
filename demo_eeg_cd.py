from args import get_best_args
from dataset import Datasets
from args import Args, get_best_args
from eeg_cd import EEG_CD_Method
from imps import *


def override_best_args(args:Args) -> Args:
    args.num_epochs = 30
    return args


src_ds, tar_ds = Datasets.SEED, Datasets.SEED_GER
alg = Algs.eeg_cd
args = get_best_args(tar_ds, alg, src_ds, key_res='res', override_fn=override_best_args)[0]
method = EEG_CD_Method()
method.run(args, fold_no=0, run=-1, echo=True)

print('finished!!!')
