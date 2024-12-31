import os
import torch

class Model_Saver:
    def __init__(self, model, args, fold_no, init_results=None, **kwargs):
        self.model = model
        self.best_res, self.best_res_train, self.best_res_test = None, None, None
        self.file_res = ''
        self.args = args
        self.fold_no = fold_no
        self.opt = kwargs['opt'] if 'opt' in kwargs.keys() else None
        self.sch = kwargs['sch'] if 'sch' in kwargs.keys() else None
        self.ema_model = kwargs['ema_model'] if 'ema_model' in kwargs.keys() else None
        self.hist = kwargs['hist'] if 'hist' in kwargs.keys() else None

        # other_save_dict = {k:v for (k,v) in kwargs.items()
        #                    if k not in ['opt', 'sch', 'ema_model', 'hist'] }
        if init_results is not None:
            if len(init_results) == 2:
                self.best_res_train, self.best_res = init_results
            else:
                self.best_res_train, self.best_res, self.best_res_test = init_results

            if self.best_res is not None:
                self.file_res = self.save_model(self.best_res, self.best_res_train,
                                                self.best_res_test, **kwargs)

    def check_save(s, res, res_train, res_test=None, **kwargs):
        if res > s.best_res:
            s.best_res, s.bestRes_train = res, res_train
            if os.path.exists(s.file_res):
                os.remove(s.file_res)
            s.file_res = s.save_model(res, res_train, res_test, **kwargs)
            return True
        else:
            return False

    def save_model(s, res, res_train, res_test=None, **kwargs):
        args = s.args
        best_score = res.score()
        ds = args.ds if type(args.ds) == str else args.ds.name
        alg = args.alg if type(args.alg) == str else args.alg.name
        res_path = f'.\\Res_{ds}'
        if not os.path.exists(res_path):
            # Create a new directory because it does not exist
            os.makedirs(res_path)
            print("The new directory is created!")

        score2 = res_test.score() if res_test is not None else res_train.score()

        if alg.lower() == 'eeg_cd':
            file_name = f'.\\Res_%s\\%s__%0.2f__%0.2f__%s__%s.mdl' % \
                        (ds, alg, best_score, score2, ds, args.src_ds.name)
        else:
            file_name = f'.\\Res_%s\\%s__%0.2f__%0.2f__%s.mdl' % \
                    (ds, alg, best_score, score2, ds)
        saved_dict = {
            'state_dict': s.model.state_dict(),
            'opt_state_dict': s.opt.state_dict() if s.opt is not None else None,
            'sch_state_dict': s.sch.state_dict() if s.sch is not None else None,
            'ema_state_dict': s.ema_model.state_dict() if s.ema_model is not None else None,
            'best_score': best_score,
            'res_train': res_train,
            'res_test': res_test,
            'res': res,
            'args': args,
            'hist': s.hist,
        }

        other_save_dict = {k:v for (k, v) in kwargs.items()
                           if k not in ['opt', 'sch', 'ema_model', 'hist'] }
        saved_dict.update(other_save_dict)
        s.other_saved_dict = other_save_dict
        torch.save(saved_dict, file_name)
        s.file_res = file_name
        return file_name

    # def _load_sd(self, name, ch):
    #
    #     if cls.sch is not None:
    #         cls.sch.load_state_dict(checkpoint['sch_state_dict'])

    def load_model(s, file_res='', return_res=False):
        if len(file_res) == 0:
            file_res = s.file_res
        checkpoint = torch.load(file_res)
        s.model.load_state_dict(checkpoint['state_dict'])
        if s.opt is not None:
            s.opt.load_state_dict(checkpoint['opt_state_dict'])
        s.args = checkpoint['args']
        if s.sch is not None:
            s.sch.load_state_dict(checkpoint['sch_state_dict'])
        if s.ema_model is not None:
            s.ema_model.load_state_dict(checkpoint['ema_state_dict'])
        s.hist = checkpoint['hist']

        s.file_res = file_res

        if return_res:
            s.best_res = checkpoint['res']
            s.best_res_train = checkpoint['res_train']
            s.best_res_test = checkpoint['res_test']
            return s.best_results

        return checkpoint

    @property
    def best_results(self):
        return [self.best_res_train, self.best_res, self.best_res_test]




