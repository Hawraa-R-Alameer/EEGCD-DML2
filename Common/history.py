from Common.Result import print_res
from Common.Evaluation import eval_clf
import matplotlib.pyplot as plt
from Common.bcolors import bcolors
import torch


class History_Base:
    def __init__(self, res, res_train, res_test=None, echo=False):
        self.test_flag = True if res_test is not None else False
        if echo:
            print('Initial Results on validation data')
            print_res(res)
            print('Initial Results on train data')
            print_res(res_train, bcolors.OKBLUE)
            if self.test_flag:
                print('Initial Results on test data')
                print_res(res_test, color=bcolors.HEADER)

        self.res_list, self.res_train_list = [], []
        if self.test_flag:
            self.res_test_list = []

    @classmethod
    def init(cls, model, train_loader, val_loader, criterion, test_loader=None, echo=False, eval_fn=None):
        if eval_fn is None:
            eval_fn = eval_clf
        res = eval_fn(model, val_loader, criterion)
        res_train = eval_fn(model, train_loader, criterion, max_ins=1000)
        if test_loader is not None:
            res_test = eval_fn(model, test_loader, criterion)
        else:
            res_test = None
        return cls(res, res_train, res_test, echo)

    def update(self, res, res_train, res_test=None):
        self.res_list.append(res)
        self.res_train_list.append(res_train)
        if self.test_flag:
            self.res_test_list.append(res_test)

    def plot_metric(self, metric='loss', metric_disp_name=''):
        if metric_disp_name == '':
            metric_disp_name = metric.capitalize()

        plt.figure()
        val_score = [res[metric] for res in self.res_list]
        train_score = [res[metric] for res in self.res_train_list]
        plt.plot(train_score)
        plt.plot(val_score)
        plt.title(metric_disp_name)
        plt.ylabel(metric_disp_name)
        plt.xlabel('Epoch')
        if self.test_flag:
            test_score = [res[metric] for res in self.res_test_list]
            plt.plot(test_score)
            plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
        else:
            plt.legend(['Train', 'Validation'], loc='upper left')

        plt.show()

    def close(self, file_res):
        update_hist_file(self, file_res)


def update_hist_file(h, file_res):
    checkpoint = torch.load(file_res)
    checkpoint['history'] = h
    torch.save(checkpoint, file_res)

