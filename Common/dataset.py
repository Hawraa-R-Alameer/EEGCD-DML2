from torch.utils.data import DataLoader, TensorDataset
from Common.utils import conv2tensor
import torch
import numpy as np
from sklearn.model_selection import train_test_split


def gen_dl(XY, batch_size=128, train=True):
    if type(XY) not in [list, tuple]:
        XY = XY, # convert XY into tuple
    XY = conv2tensor(*XY)
    ds_train = TensorDataset(*XY)
    shuffle = True if train else False
    ds_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    return ds_loader


def gen_dls(data, args):
    bs = args.get('batch_size', 128)
    eval_bs = args.get('eval_batch_size', bs)

    train_loader = gen_dl((data.X, data.y), train=True,  batch_size=bs)
    val_loader = gen_dl((data.X_val, data.y_val), train=False,  batch_size=eval_bs)
    if torch.equal(data.y_val, data.y_test):
        test_loader = None
    else:
        test_loader = gen_dl((data.X_test, data.y_test), train=False, batch_size=eval_bs)

    return train_loader, val_loader, test_loader


def cls_distribution(y):
    if torch.is_tensor(y):
        return torch.round(100 * y.bincount() / len(y), decimals=2)
    return np.round(100 * np.bincount(y) / len(y), 2)


def get_Xy(df, target):
    y = df[target].values.astype('int64')
    X = df.drop([target], axis=1).values.astype('float32')
    return X, y


def split_data(data, seed=0, val_ratio=0.2, test_ratio=.2, stratify=True):
    n = len(data.y)
    ind = list(range(n))
    if test_ratio > 0:
        ind_tr_val, ind_test = train_test_split(ind, test_size=test_ratio, random_state=seed,
                                                stratify=data.y if stratify else None)
        data.X_test, data.y_test = data.X[ind_test], data.y[ind_test]
    else:
        ind_tr_val = ind
        ind_test = None

    if val_ratio > 0:
        ind_tr, ind_val = train_test_split(ind_tr_val, test_size=val_ratio, random_state=seed,
                                           stratify=data.y[ind_tr_val] if stratify else None)
        data.X_val, data.y_val = data.X[ind_val], data.y[ind_val]
    else:
        ind_tr = ind_tr_val
        ind_val = None

    data.X, data.y = data.X[ind_tr], data.y[ind_tr]
    data.idx_train, data.idx_val, data.idx_test = ind_tr, ind_val, ind_test
    return data


def sample_ds(X, y, sample_ratio, min_ins=1000, seed=0):
    n_classes = np.bincount(y)
    # n_c = len(n_classes)
    min_classes = np.where(n_classes <= min_ins)[0]
    # other_classes = [c for c in range(n_c) if c not in min_classes]
    mask = np.isin(y, min_classes)
    sel_idx = np.where(mask)[0]
    other_classes_idx = np.where(~mask)[0]
    other_sel_idx, _ = train_test_split(other_classes_idx, test_size=1 - sample_ratio,
                                        random_state=seed, stratify=y[other_classes_idx])
    sel_idx = np.concatenate((sel_idx, other_sel_idx))
    X, y = X[sel_idx], y[sel_idx]
    return X, y, sel_idx


def main():
    import numpy as np
    X = np.random.random((150,5))
    dl = gen_dl(X)
    y = np.random.randint(0, 2, 13)
    print('finished!!!')


if __name__ == '__main__':
    main()
    print('Congratulations to you!')