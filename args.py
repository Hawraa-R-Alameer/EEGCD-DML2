from utils import *
from dataset import Datasets
from _imps import *


class Args(MyDict):
    def __init__(s, ds:Datasets, alg=Algs.eeg_dml, dim=Emotion_Dim.valence): # s:self
        super(Args, s).__init__()
        s.seed = 42
        s.n_channels = 32 if ds == Datasets.DEAP else 62
        from dataset import ds_info
        s.window_size = ds_info[ds.name].window_size
        s.n_subj = ds_info[ds.name].n_subj
        s.batch_size = ds_info[ds.name].bs

        # s.window_size = 60*4 if ds == Datasets.DEAP else 265*5
        # if ds == Datasets.SEED_GER:
        #     s.window_size = 60*5

        s.n_classes = 3
        if ds in [Datasets.SEED, Datasets.SEED_GER]:
            s.dimension = Emotion_Dim.discrete
        else:
            s.dimension = dim

        s.dropout_rate = 0.1
        # s.mean_w, s.sigma_w = 0, .1  # variance for weak augmentation
        s.lr = 1e-05
        # s.aug = Aug.both if alg == Algs.eeg_dml else Aug.none #'both'
        s.aug = None
        s.num_epochs = 100
        s.patient = 50
        s.opt, s.weight_decay = Optim.adam , 0.0001
        s.sch =  Sch.CosineAnnealingLR #Sch.none # CosineAnnealingLR/StepLR/None
        s.ds, s.alg = ds, alg
        s.d_e = 128 # embed dim

        if s.alg == Algs.clf_adv:
            s.loss_parts = ['loss', 'l_c', 'l_d']
            s.lam = .5
        elif s.alg == Algs.eeg_dml:
            # s.lam = 0.
            # s.lam_s = .5
            s.K = 5
            s.gamma = 1
            s.dml_flag = True
            s.pre_train = False
            s.la = 20
            s.center_lr = 0.0005
            s.mix_up_rate = 0.
        elif s.alg == Algs.eeg_cd:
            s.src_ds = Datasets.SEED
            s.n_subj = ds_info[s.src_ds.name].n_subj
            s.tar_ds = ds
            s.K = 5
            s.gamma = 1
            s.la = 20
            s.center_lr = s.lr*10 #
            s.lam_subj = 1
            s.lam_dom = 1
            # strong aug parameters
            s.xi, s.eps, s.ip = 10.0, 3.0, 1
            s.distribution_matching = True
            s.relative_confidence = True
            s.threshold = .95 # confidence threshold
            s.T = .5 # softmax temprature


        s.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def check(args, ds, alg):
        def_args = Args(ds, alg)
        for k in def_args.keys():
            if k not in args:
                args[k] = def_args[k]

        return args

class Args_Dict(MyDict):
    def __init__(s): # s:self
        super().__init__()
        s.lr = [1e-5, 3e-5, 5e-5, 1e-4]
        s.batch_size = [16, 32]
        s.K = [1, 3, 5]
        s.center_lr = [1e-4, 5e-4, .001, 3e-3, .005, .01]
        s.gamma = [.1, 1]
        s.lam_s = [.1, .5, 1]
        s.mix_up_rate = [0., .1, .2, .3]




def get_best_args(ds, alg, src_ds=None, override_fn=None, echo=True, path_res=None, file_name='',
                  key_res='res_val')-> tuple[Args, Result]:
    args = Args(ds=ds, alg=alg)
    seed_all(seed=args.seed)
    best_score, p, best_res, hist_best = get_bestRes(ds.name, alg.name, src_ds, path_res,
                                                         file_name, key_res=key_res)

    if len(file_name) > 0:
        p.ds, p.alg = ds, alg

    print(bcolors.OKBLUE + 'best Score=%.2f' % best_score + bcolors.ENDC)
    if best_score == 0:
        print('best args not found!!!')
    else:
        args = p
        args = Args.check(args, ds, alg)
        args.print(name='best existing args')
        if len(file_name) > 0:
            from utils import disp_res
            disp_res(file_name)
        else:
            print_res(best_res, summary_sw=True)

    args = override_fn(args) if override_fn is not None else args
    args.print('args:')
    seed_all(seed=args.seed)
    return args, best_res

def main():
    from dataset import Datasets
    # ds = Datasets.SEED
    # alg = Algs.eeg_dml
    # args, best_res = get_best_args(ds, alg)
    args_ = get_best_args(Datasets.SEED_GER, Algs.eeg_cd, Datasets.SEED)
    print(args_)
    pass


if __name__ == '__main__':
    main()
    print('Congratulations to you!')
