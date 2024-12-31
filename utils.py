from torch.autograd import Function
from common.history import History_Base
from imps import *


class Emotion_Dim(IntEnum):
    valence = 0
    arousal = 1
    discrete = 2



class Optim(IntEnum):
    adam = 0
    adamW = 1
    rms_prop = 2
    ada_delta = 3
    sgd = 4


class Sch(IntEnum):
    CosineAnnealingLR = 0
    StepLR = 1
    none = 2


def create_model(args):
    create_model_fn = None
    alg: Algs = args.alg
    match alg:
        case Algs.eeg_dml:
            from eeg_dml import create_model
            create_model_fn = create_model
        case _:
            print("Invalid Algorithm.")

    model = create_model_fn(args)
    return model


def distribution_match(pseudo_label, preds_s):
    mean_preds_s = torch.mean(preds_s, dim=0)
    mean_preds_u = torch.mean(pseudo_label, dim=0)
    ratio = (1e-6 + mean_preds_s) / (1e-6 + mean_preds_u.detach())
    pseudo_label *= ratio
    pseudo_label = pseudo_label / pseudo_label.sum(dim=1, keepdim=True)
    return pseudo_label



def create_opt_sch(grouped_params, args):
    match args.opt:
        case Optim.adam:
            opt = torch.optim.Adam(grouped_params, lr=args.lr,
                                   weight_decay=args.weight_decay)
        case Optim.adamW:
            opt = torch.optim.AdamW(grouped_params,
                                    lr=args.lr, weight_decay=args.weight_decay)
        case Optim.sgd:
            opt = torch.optim.SGD(grouped_params,
                                  lr=args.lr, momentum=0.9, nesterov=args.nesterov)
        case Optim.ada_delta:
            opt = torch.optim.Adadelta(grouped_params,
                                       lr=args.lr, weight_decay=args.weight_decay)
        case Optim.rms_prop:
            opt = torch.optim.Adadelta(grouped_params,
                                       lr=args.lr, weight_decay=args.weight_decay)
        case _:
            print(bcolors.WARNING + 'optimizer not recognized!!!, '
                                    'default optimizer (adam) is set' + bcolors.ENDC)
            opt = torch.optim.Adam(grouped_params, lr=args.lr, weight_decay=args.weight_decay)

    match args.sch:
        case Sch.CosineAnnealingLR:
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10, eta_min=0)
        case Sch.StepLR:
            sch = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.9)
        case _:
            sch = None

    return opt, sch


def conv2tensor(*l):
    return [X if torch.is_tensor(X) else torch.from_numpy(X) for X in l]


def to_categorical(y, num_classes=None, dtype='float32'):
    #one-hot encoding
    y = np.array(y, dtype='int16')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, _lambda):
        ctx.save_for_backward(x, _lambda)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, _lambda = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - _lambda * grad_output
        return grad_input, None


revgrad = GradientReversal.apply

class GRL(nn.Module):
    def __init__(self, lam):
        super().__init__()
        self.lam = torch.tensor(lam, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.lam)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_bestRes(ds, alg, src_ds =None, path_res=None, file_name='', key_res='res',
                whole_checkpoint=False):
    if(path_res is None):
        path_res = '.\\Res_'

    dir_name = path_res + ds
    if len(file_name) > 0:
        file_path = os.path.join(dir_name, file_name)
        return load_res(file_path, key_res=key_res, whole_checkpoint=whole_checkpoint)

    # best_score, bestParams, best_resInfo, hist_best = 0, [], [], []
    best_score, best_args, best_res, best_hist = 0, None, None, None
    best_ch = None

    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        if whole_checkpoint:
            return best_score, best_args, best_res, best_hist, best_ch
        else:
            return best_score, best_args, best_res, best_hist,

    file_names = os.listdir(dir_name)
    for f in file_names:
        if not (f.endswith('cvr') or f.endswith('mdl')):
            continue
        m = f.lower().split('__') # m[0] shows model name
        if len(m) > 0 and m[0] == alg.lower():
            try:
                ch = torch.load(os.path.join(dir_name, f))
                args = ch['args']
                if (src_ds is not None) and (src_ds != args.src_ds):
                    continue
                if key_res == 'res' and 'res' not in ch.keys():
                    key_res = 'res_val'
                res = ch[key_res]
                hist = ch.get('history', None)
                score = res.score() #ch['best_score'],
                if score > best_score:
                    best_score = score
                    best_args = ch["args"]
                    best_res = res
                    best_hist = hist
                    best_args.file_name = os.path.join(dir_name, f)
                    best_ch = ch

            except BaseException as err:
                print(f"Unexpected {err=}, {type(err)=}")

    if whole_checkpoint:
        return best_score, best_args, best_res, best_hist, best_ch
    else:
        return best_score, best_args, best_res, best_hist


class Method:
    def __init__(self, args):
        self.args = args
        self.K = args.n_subj
        self.pre_train_flag = args.pre_train
    def end_criterion(self, hist:History_Base):
        res_train = hist.res_train_list[-1]
        # if res_train.score() > 92.:
        if res_train.acc > 97.0 and res_train.roc_auc > 99.0:
            return True
        else:
            return False

    def pre_train(self, model, dataset, criterion=None, plot_flag=False):
        from common.Result import AverageMeter
        from losses import InfoNCELoss, BatchSampler, Projector
        proj = Projector(model.d_f, output_dim=128).cuda()
        criterion = InfoNCELoss(temperature=.1) if criterion is None else criterion
        from torch.utils.data import DataLoader
        batch_sampler = BatchSampler(dataset)
        data_loader = DataLoader(dataset, batch_sampler=batch_sampler)
        opt = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0001)
        epochs_cl = 5
        for epoch in range(epochs_cl):
            loss_am = AverageMeter()
            from tqdm import tqdm
            p_bar = tqdm(range(len(data_loader)))
            features, labels = [], []
            for batch in data_loader:
                X = batch[0][0].cuda()
                y = batch[1].cuda()
                opt.zero_grad()
                model.train()
                X_f = model(X, return_fea=True)
                X_e = proj(X_f)
                loss = criterion(X_e[0], X_e[1], X_e[2:])  # anchor, positive, and negatives
                loss.backward()
                opt.step()
                loss_am.update(loss.item())
                p_bar.set_postfix_str(f'loss={loss.item():.4f} loss_avg:{loss_am.avg:.4f}')
                features.append(X_e.cpu().detach())
                labels.append(y.cpu().detach())
                p_bar.update()
            print(f'epoch:{epoch + 1}/{epochs_cl}, loss={loss_am.avg:.4f}')
            p_bar.close()

        # if plot_flag:
        #     proxies = criterion_s.W.T.cpu().detach()
        #     # Concatenate the lists into tensors
        #     features = torch.cat(features, dim=0)
        #     labels = torch.cat(labels, dim=0)
        #     plot_features_and_proxies(features, labels, proxies)

        torch.cuda.empty_cache()

    def run(self, fold_no=0, run=-1, echo=True):
        from dataset import gen_dl
        from trainer import Trainer
        args = self.args
        train_loader, test_loader = gen_dl(args.ds, self.K, fold_no, args.batch_size,
                                           mixup_rate=args.mix_up_rate)
        model = create_model(args)
        group_params = model.group_params()
        opt, sch = create_opt_sch(group_params, args)
        seed_all(seed=args.seed)
        if self.pre_train_flag:
            self.pre_train(model, train_loader.dataset)

        trainer = Trainer(model, opt, args, sch, end_criterion_fn=self.end_criterion,
                          fold=fold_no, run=run)
        trainer.fit(train_loader, val_loader=test_loader, test_loader=None)
        torch.cuda.empty_cache()
        from dataset import Datasets
        if args.ds == Datasets.SEED_GER:
            res_train, res_test = trainer.best_res_train, trainer.best_res
        else:
            res_train, res_test = trainer.last_results[0], trainer.last_results[1]
        if echo:
            pre_text = '' if run==-1 else f'run={run}, '
            pre_text += f'fold_no={fold_no} '
            # res_test.print(color=bcolors.FAIL, pre_text=pre_text + 'Test Results:')
            print_res(res_test, color=bcolors.HEADER, pre_text=pre_text + 'Test Results:',
                      summary_sw=True)
            print_res(res_train, color=bcolors.OKBLUE, pre_text=pre_text + 'Train Results:',
                      summary_sw=True)

        torch.cuda.empty_cache()
        self.dip_cuda_usage()
        return res_train, res_test, trainer.hist

    def dip_cuda_usage(self):
        device = 'cuda'
        free = torch.cuda.mem_get_info()[0] / (1024 ** 2)
        used = torch.cuda.memory_allocated(device) / (1024 ** 2)
        print(f'Cuda free:{free} MB, used:{used} MB')


def main():
    from dataset import Datasets
    disp_res('./Res_SEED/eeg_dml__91.09__97.17__SEED.cvr')
    ds = Datasets.SEED
    alg = Algs.eeg_dml
    from args import get_best_args
    args = get_best_args(ds, alg, echo=True)[0]
    args.pre_train = True
    method = Method(args)
    method.run(fold_no=0, run=-1, echo=True)
    pass


if __name__ == '__main__':
    main()
    print('Congratulations to you!')

