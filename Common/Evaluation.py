import torch
import math
import numpy as np
from common.Struct import Struct
from common.Result import Result
from common.classification_res import report_bin_classification, report_multi_classification
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.spatial.distance import cdist
from statistics import mode
from tqdm import tqdm
import pickle
from os import path


def eval_clf(model, data_loader, criterion, device='cuda', epoch=-1, max_ins=math.inf, run_model_fn=None):
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=-1)
        model.eval()
        y_pred, probs = torch.Tensor(), torch.Tensor()
        y_true = torch.Tensor()
        loss = torch.tensor(0.0).to(device)
        for data, labels in data_loader:
            if isinstance(data, list):
                data = [item.to(device) if len(item) > 0 else item for item in data]
            else:
                data = data.to(device)
            labels = labels.long().to(device)
            out = model(data) if run_model_fn is None else run_model_fn(data)
            if isinstance(out, tuple) or isinstance(out, list):
                out = out[0]
            preds = out.argmax(-1)
            l = criterion(out, labels)
            if isinstance(out, tuple) or isinstance(out, list):
                l = l[0]
            loss += l.item() * labels.size(0)
            y_pred = torch.cat((y_pred, preds.cpu()))
            probs = torch.cat((probs, softmax(out.cpu())))
            y_true = torch.cat((y_true, labels.cpu()))
            if len(y_true) >= max_ins:
                break

    loss = loss / len(y_pred)
    y_pred, y_true = y_pred.numpy(), y_true.numpy()
    if len(np.unique(y_true)) == 2:
        res = report_bin_classification(y_true, y_pred, probs, percent=True)
    else:
        res = report_multi_classification(y_true, y_pred, probs, percent=True)

    res.loss, res.epoch = loss.item(), epoch
    return res


def calc_map(y, y_NN, K=None):
    if K is None:
        K = y_NN.shape[1]

    n= len(y)
    related = y.reshape((n, 1)) == y_NN[:,:K]
    num_related = np.sum(related, axis=1, keepdims=True)
    num_related[num_related == 0] = 1
    cs = np.cumsum(related, axis=-1)
    recall = cs / num_related
    precision = cs / np.arange(1, K + 1)
    avg_p = np.sum(precision * related, axis=1, keepdims=True) / num_related
    map = np.mean(avg_p, axis=0)
    precision_K = np.mean(precision[:,-1])
    return map[0]*100, precision_K*100


def evaluation(X, y, Kset=None, X_ref=None, y_ref=None, kNN_K = 3, K=None):
    if Kset is None:
        Kset = [1, 2, 4, 8]
    res = Struct()
    num_classes = len(np.unique(y))
    K = K if K is not None else np.max(Kset)
    # kmax = np.max(Kset)
    res.recallK = np.zeros(len(Kset))
    if X_ref is None:
        X_ref, y_ref = X, y

    n = len(X)

    #compute Recall@K and MAP
    # mat = cdist(X, X_ref, 'hamming')
    q = X.shape[-1]
    mat = .5 * (q - np.dot(X, X_ref.transpose()))
    indices = np.argsort(mat, axis=1)[:, 1:K+1]

    y_NN = y_ref[indices]
    y_pred = np.zeros(shape=(n), dtype=int)
    for j in range(0, n):
        y_pred[j] = mode(y_NN[j, :kNN_K])

    res.acc_kNN = np.sum(y_pred == y) / n * 100
    res.K = K
    res.map, res.precision_K = calc_map(y, y_NN)

    # Recall@k is the  proportion of relevant items found in the top - k retrievd items
    for i in range(0, len(Kset)):
        pos = 0.
        for j in range(0, n):
            if y[j] in y_NN[j, :Kset[i]]:
                pos += 1.
        res.recallK[i] = pos/n * 100
    return res, y_pred


def eval_model(model, data_loader, criterion, save_hash = None, device='cuda', max_ins=math.inf, type='test',
               CBIR_flag=True):
    with torch.no_grad():
        model.eval()
        y_pred, out_h = torch.Tensor(), torch.Tensor()
        y_true = torch.Tensor()
        total, correct = 0, 0
        loss = torch.tensor(0.0).to(device)
        with tqdm(total=len(data_loader)) as t:
            for (data, labels) in data_loader:
                data, labels = data.to(device), labels.to(device)
                if save_hash is None:
                    out, h = model(data, return_hash=2)
                else:
                    out = model(data)
                    h = save_hash.outputs[0]
                    save_hash.clear()

                l = criterion(out, labels)
                loss += l * labels.size(0)
                pred = out.argmax(1)
                correct += pred.eq(labels).sum().item()
                y_pred = torch.cat((y_pred, pred.cpu()))
                out_h = torch.cat((out_h, h.cpu()))
                y_true = torch.cat((y_true, labels.cpu()))
                total += labels.size(0)
                if len(y_true) >= max_ins:
                    break
                # t.set_postfix(loss='{:05.3f}'.format(loss / total), acc='%2f' % (100. * correct / total))
                t.set_postfix({type + ' loss':'{:05.3f}'.format(loss/total), type + ' acc':'%2f' % (100.*correct/total)})
                t.update()


    loss = loss / len(y_pred)
    y_pred, out_h, y_true = y_pred.numpy(), out_h.numpy(), y_true.numpy()
    acc = np.sum(y_pred == y_true) * 100. / len(y_pred)
    if CBIR_flag:
        res, _ = evaluation(out_h, y_true)
    else:
        res = Struct()
    res.acc, res.loss = acc, loss

    return res, y_pred, y_true


def init_history(model, train_loader, test_loader, criterion, device='cuda', save_hash = None):
    res_train, _, _ = eval_model(model, train_loader, criterion, device=device, CBIR_flag=False, type='train',
                           save_hash=save_hash)
    res, _, _ = eval_model(model, test_loader, criterion, device=device, save_hash=save_hash)
    h = Struct()
    h.acc_train,h.loss_train = [res_train.acc], [res_train.loss.item()]
    h.acc, h.loss, h.map = [res.acc], [res.loss.item()], [res.map]
    h.recall, h.acc_kNN = [res.recallK], [res.acc_kNN]

    return h

def update_history(h, res, acc_train, loss_train):
    h.acc_train.append(acc_train), h.loss_train.append(loss_train)
    h.acc.append(res.acc), h.loss.append(res.loss.item()), h.map.append(res.map)
    h.recall.append(res.recallK), h.acc_kNN.append(res.acc_kNN)
    return h


def update_hist_result(h, file_res):
    if file_res == '':
        return False

    with open(file_res, 'rb') as file:
        score, params, res, _ = pickle.load(file)

    with open(file_res, 'wb') as file:
        pickle.dump([score, params, res, h], file)

    return True




