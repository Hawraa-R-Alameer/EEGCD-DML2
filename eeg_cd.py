import filecmp
import seaborn
import torch
from vat import VATLoss
from sognn import SOGNN_DSBN
from utils import GRL, create_opt_sch, distribution_match, seed_all
from args import Args
from losses import Emotion_Loss, Projector
from sklearn.metrics import roc_auc_score
from _imps import *
from args import Args
from dataset import Datasets, gen_cd_dls


class EEG_CD(nn.Module):
    def __init__(s, args:Args):
        super().__init__()
        s.args = args
        s.device = args.device
        s.fea_module = SOGNN_DSBN()
        s.d_f = s.fea_module.linend.in_features
        s.d_e = args.get('d_e', 128)
        s.proj = Projector(s.d_f, output_dim=s.d_e).to(s.device)
        s.criterion = Emotion_Loss(n_classes=args.n_classes, K=args.K,
                                     gamma=args.gamma,
                                     embed_dim=s.d_e).to(s.device)
        s.lam_subj = args.lam_subj
        s.lam_dom = args.lam_dom
        s.n_subj = args.n_subj
        s.subj_dis = s.create_subj_dis()
        s.dom_dis = s.create_dom_dis()
        s.to(s.device)
        s.vat_loss = VATLoss(xi=s.args.xi, eps=s.args.eps, ip=s.args.ip)

    def create_subj_dis(s):
        d = s.d_e # d_f
        model = nn.Sequential(GRL(s.lam_subj),
                      nn.Linear(d, d * 2), nn.ReLU(),
                      # nn.LeakyReLU(0.2, inplace=True),
                      nn.Linear(d * 2, d), nn.ReLU(),
                      # nn.LeakyReLU(0.2, inplace=True),
                      nn.Linear(d, s.n_subj))
        return model

    def create_dom_dis(s):
        d = s.d_e # d_f
        model = nn.Sequential(GRL(s.lam_dom),
                   nn.Linear(d, d * 2), nn.ReLU(),
                   # nn.LeakyReLU(0.2, inplace=True),
                   nn.Linear(d * 2, d), nn.ReLU(),
                   # nn.LeakyReLU(0.2, inplace=True),
                   nn.Linear(d, 2))
        return model


    # def forward(self, x: torch.Tensor, mask, return_fea=False) -> torch.Tensor:
    def forward(self, x: torch.Tensor, domain='src') -> torch.Tensor:
        x = self.fea_module(x, domain=domain, return_fea=True)
        return x

    def split_batch(self, batch):
        inputs, targets = batch[0], batch[1]
        targets = targets.to(self.device)
        X, l = inputs[0].to(self.device), inputs[1].to(self.device)
        return X, l, targets.long()

    def train_step(s, batch_s, batch_t, return_out=False):
        x_src, l_src, y_src = s.split_batch(batch_s)
        x_tar_w, l_tar, y_tar = s.split_batch(batch_t)
        n_tar, n_src = len(x_tar_w), len(x_src)

        fea_src = s.fea_module(x_src, domain='src')
        # compute loss src
        emb_src = s.proj(fea_src)
        loss_src, preds_src = s.criterion(emb_src, y_src)
        preds_src = torch.softmax(preds_src/s.args.T, dim=-1)

        # compute loss subj
        out_subj = s.subj_dis(emb_src)
        loss_subj = F.cross_entropy(out_subj, l_src)

        # x_src = x_src.detach()
        # s.opt.zero_grad()
        # # compute loss again
        # fea_src = s.fea_module(x_src, domain='src')
        # emb_src = s.proj(fea_src)
        # l_src = s.criterion(emb_src, y_src)[0]
        # l_src.backward()
        # s.opt.step()
        # print(l_src)

        # 2. compute loss target

        # 2.1) x_tar_s = obtain strong aug
        lds, r_adv = s.vat_loss(s.fea_module, x_tar_w)
        x_tar_s = x_tar_w + r_adv

        # 2.2) generate pseudo-labels
        fea_tar = s.fea_module(torch.cat([x_tar_w, x_tar_s]), domain='tar')
        emb_tar = s.proj(fea_tar)
        # fea_tar_w, fea_tar_s = torch.chunk(fea_tar, 2)
        emb_tar_w, emb_tar_s = torch.chunk(emb_tar, 2)
        p_tar_w = s.criterion(emb_tar_w, targets=None)
        pseudo_label = torch.softmax(p_tar_w.clone().detach()/s.args.T, dim=-1)
        if s.args.distribution_matching:
            pseudo_label = distribution_match(pseudo_label, preds_s=preds_src)

        # 2.3 compute loss tar
        max_probs, targets_tar = torch.max(pseudo_label, dim=-1)
        if s.args.relative_confidence:
            confidence_coeff = preds_src.max(dim=1)[0].mean(0)
        else:
            confidence_coeff = 1.
        mask_b = max_probs.ge(confidence_coeff * s.args.threshold)
        mask = mask_b.float()
        loss_tar_ins, p_tar_s = s.criterion(emb_tar_s, targets_tar, reduction='none')
        loss_tar = 1/n_tar * (loss_tar_ins * mask).sum()

        # create loss domain
        # y_dom = create domain target
        y_dom = torch.cat([torch.zeros(n_src, device=s.device,),
                           torch.ones(n_tar, device=s.device)]).long()
        emb_dom = torch.cat([emb_src, emb_tar_w])
        order = torch.randperm(len(y_dom))
        y_dom, emb_dom = y_dom[order], emb_dom[order]
        out_dom = s.dom_dis(emb_dom)
        loss_dom = F.cross_entropy(out_dom, y_dom)
        loss = (loss_src + loss_tar + s.lam_subj * loss_subj +
                s.lam_dom * loss_dom)
        loss_dict = MyDict(loss_src=loss_src, loss_tar=loss_tar,
                           loss_subj=loss_subj, loss_dom=loss_dom, loss=loss)
        acc_tar = 100*(y_tar == targets_tar).sum() / len(y_tar)
        y_pred_src = torch.argmax(preds_src, -1)
        acc_src = 100*(y_src == y_pred_src).sum() / len(y_src)
        other_dict = MyDict(mask=100*mask.mean(), conf=100*max_probs.mean(),
                            acc_tar=acc_tar, acc_src=acc_src)
        return loss_dict, other_dict

    def load_batch_tar(self):
        try:
            batch = next(self.tar_iter)
        except:
            self.tar_iter = iter(self.tar_loader)
            batch = next(self.tar_iter)
        # x, l, y = self.split_batch(batch)
        return batch


    def train_eval(s, src_loader, tar_loader, test_loader, best_score=0, fold_no=-1, run=-1):
        # create opt, sch
        group_params = s.group_params()
        s.opt, s.sch = create_opt_sch(group_params, s.args)

        # evaluate model before train
        res, res_test = s.eval_model(tar_loader, test_loader, epoch=0, fold_no=fold_no,
                                     run=run)
        best_score = max([best_score, res.score()])
        # initialize history
        hist = History_Base(res, res, res_test)
        best_res, best_res_test = res, res_test
        s.tar_iter = iter(tar_loader)
        s.tar_loader = tar_loader
        sch_flag = s.sch is not None
        from Model_Saver import Model_Saver
        model_saver = Model_Saver(s, s.args, fold_no, init_results=(res, res, res_test))
        p = 0
        patient = s.args.patient if 'patient' in s.args else s.args.num_epochs // 3
        epoch = 1
        while epoch <= s.args.num_epochs:
            # create average_meters
            loss_am, loss_src_am, loss_tar_am = AverageMeter(), AverageMeter(), AverageMeter()
            loss_subj_am, loss_dom_am = AverageMeter(), AverageMeter()
            acc_tar_am, mask_am, conf_am = AverageMeter(), AverageMeter(), AverageMeter()
            acc_src_am = AverageMeter()

            p_bar = tqdm(range(len(src_loader)))
            for batch_s in src_loader:
                # batch_t = load a batch from tar_loader using tar_iter
                batch_t = s.load_batch_tar()
                s.opt.zero_grad()
                loss_dict, other_dict = s.train_step(batch_s, batch_t)
                loss = loss_dict.loss
                loss.backward()
                s.opt.step()
                if sch_flag:
                    s.sch.step()
                loss_am.update(loss.item())
                loss_src_am.update(loss_dict.loss_src.item())
                loss_tar_am.update(loss_dict.loss_tar.item())
                loss_subj_am.update(loss_dict.loss_subj.item())
                loss_dom_am.update(loss_dict.loss_dom.item())
                acc_tar_am.update(other_dict.acc_tar.item())
                mask_am.update(other_dict.mask.item())
                conf_am.update(other_dict.conf.item())
                acc_src_am.update(other_dict.acc_src.item())

                lr = s.sch.get_last_lr()[0] if sch_flag else s.opt.defaults['lr']

                info = (f'Epoch:{epoch}/{s.args.num_epochs}, '
                        f'loss={loss_am.avg:.4f}, l_src={loss_src_am.avg:.3f}, '
                        f'acc_tar={acc_tar_am.avg:.2f}, acc_src={acc_src_am.avg: .2f}, '
                        f'mask={mask_am.avg:.2f}, conf={conf_am.avg:2f}, '
                        f'l_tar={loss_tar_am.avg:.3f}, l_src={loss_src_am.avg:.3f}, '
                        f'l_subj={loss_subj_am.avg:.3f}, l_dom={loss_dom_am.avg:.4f}'                       
                        f' LR:{lr:.6f}')

                p_bar.set_postfix_str(info)
                p_bar.update()
            epoch += 1
            # evaluate model
            res, res_test = s.eval_model(tar_loader, test_loader, epoch=epoch,
                                         fold_no=fold_no, run=run)
            hist.update(res, res, res_test)
            if res.score() > best_score:
                best_res, best_res_test = res, res_test
                model_saver.save_model(res, res, res_test)
                p = 0
            else:
                p += 1
                if p == patient:
                    break

        s.file_res = model_saver.file_res
        # load best model
        s.best_res, s.best_res_test = best_res, best_res_test
        model_saver.load_model(return_res=False)
        hist.close(s.file_res)
        print('training steps finishded!!!')
        return best_res, best_res_test


    def eval_model(s, tar_loader, test_loader=None, epoch=0, fold_no=-1, run=-1):
        pre_text = 'Initial ' if epoch == 0 else ''
        if run >= 0:
            pre_text += f'run no:{run} '
        if fold_no >= 0:
            pre_text += f'fold no:{fold_no} '

        res_tar = s.evaluate(tar_loader, epoch=epoch, set='target')
        if epoch > 0 and (res_tar > s.best_res):
            print('target score=', res_tar.score(), 'a better result is obtained!!!')
            color = bcolors.BOLD
        else:
            color = bcolors.FAIL
        print_res(res_tar, color=color, pre_text=pre_text + 'Target Results', summary_sw=True)
        if test_loader is not None:
                res_test = s.evaluate(test_loader, epoch=epoch, set='test')
                print_res(res_test, color=bcolors.HEADER, pre_text=pre_text + 'Test Results',
                      summary_sw=True)
        else:
            res_test = None

        if epoch == 0:
            s.best_res, s.best_res_test = res_tar, res_test

        return res_tar, res_test

    def evaluate(self, data_loader, max_ins=math.inf, **kwargs):
        from tqdm import tqdm
        p_bar = tqdm(range(len(data_loader)))
        with torch.no_grad():
            self.eval()
            probs, preds, y, loss = [], [], [], []
            count = 0
            for batch in data_loader:
                x, l, target_batch = self.split_batch(batch)
                x_f = self(x, domain='tar')
                x_e = self.proj(x_f)
                l_s, out_s = self.criterion(x_e, target_batch)
                prob_b = torch.softmax(out_s, dim=1).detach().cpu()
                pred_b = torch.argmax(out_s, -1).detach().cpu()
                probs.append(prob_b)
                preds.append(pred_b)
                loss.append(l_s.item())

                y.append(target_batch.detach().cpu())
                p_bar.update()
                count += len(target_batch)
                p_bar.update()
                if count >= max_ins:
                    break

        p_bar.close()
        y = torch.cat(y).numpy()

        from utils import to_categorical
        targets = to_categorical(y)
        probs = torch.cat(probs).numpy()
        preds = torch.cat(preds).numpy()
        res_s = report_multi_classification(y, preds, probs)
        res_s.roc_auc = roc_auc_score(targets, probs, average='macro') * 100
        res_s.loss = np.array(loss).mean()
        for key, val in kwargs.items():
            res_s[key] = val

        return res_s

    def group_params(self):
        params = set(self.parameters())
        params = list(params.difference(set(self.criterion.parameters())))
        group_params = [{"params": params, "lr":self.args.lr},
                        {"params": self.criterion.parameters(),
                         "lr": self.args.center_lr}
                        ]
        return group_params

    def print_parameters(self):
        i = 0
        for n, p in self.named_parameters():
            print(f'{i}: {n}, ', p.shape)
            i += 1


def get_score(res):
    return res.acc

def create_model(args) -> EEG_CD:
    # create model
    model = EEG_CD(args)
    model.zero_grad()
    return model


class EEG_CD_Method:
    def __init__(self):
        pass

    def run(self, args, fold_no=0, run=-1, echo=True):
        seed_all(seed=args.seed)
        src_loader, tar_loader, test_loader = gen_cd_dls(src_ds=args.src_ds,
                                                         tar_ds=args.tar_ds, subj_test=fold_no)
        eeg_cd = create_model(args)
        res, res_test = eeg_cd.train_eval(src_loader, tar_loader, test_loader, fold_no=fold_no,
                                          run=run)
        torch.cuda.empty_cache()
        if echo:
            pre_text = '' if run==-1 else f'run={run}, '
            pre_text += f'fold_no={fold_no} '
            print_res(res, color=bcolors.FAIL, pre_text=pre_text + 'Target Results:',
                      summary_sw=True)
            print_res(res_test, color=bcolors.HEADER, pre_text=pre_text + 'Test Results:',
                      summary_sw=True)

        torch.cuda.empty_cache()
        self.dip_cuda_usage()
        return res, res_test

    def dip_cuda_usage(self):
        device = 'cuda'
        free = torch.cuda.mem_get_info()[0] / (1024 ** 2)
        used = torch.cuda.memory_allocated(device) / (1024 ** 2)
        print(f'Cuda free:{free} MB, used:{used} MB')

def main():
   print('finished!!!')


if __name__ == '__main__':
    main()
    print('Congratulations to you!')