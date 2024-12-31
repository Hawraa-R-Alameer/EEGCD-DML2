import numpy as np

version = 1
import scipy.io as sio
import glob
from utils import Emotion_Dim
from imps import *

class Datasets(IntEnum):
    DEAP = 0
    SEED = 1
    SEED_GER = 2
    MPED = 3
    SEED4 = 4



ds_info = MyDict(
    SEED_GER=MyDict(window_size=265*5, bs=16, n_subj=8, ds_path='./ds/seed_ger.ds'),
    SEED=MyDict(window_size=265*5, bs=16, n_subj=15, ds_path='./ds/seed.ds'),
    DEAP=MyDict(window_size=60*4, bs=16, n_subj=32),
    SEED4=MyDict(window_size=265 * 5, bs=16, n_subj=15, ds_path='./ds/seed4.ds'),
)


class AugDataset(Dataset):
    def __init__(self, X, subject_labels, y, stage='train', aug_fn=None, args_aug=None,
                 return_index=False, mix_up_rate=.2):
        self.stage = stage
        self.X = X.float()
        self.y = y
        self.l = subject_labels
        self.subjects = subject_labels.unique()
        self.map2node = {k.item():v for (v, k) in enumerate(self.subjects)}
        self.l_targets = torch.tensor([self.map2node[subj.item()] for subj in self.l])
        self.fn_aug = aug_fn
        self.args_aug = args_aug
        self.return_index = return_index
        self.mix_up_rate = mix_up_rate
        self.class_ind = dict()
        self.alpha: float = 0.2
        self.classes = self.y.unique()
        for c in self.classes:
            self.class_ind[c.item()] = torch.where(self.y == c)[0].tolist()

    def __getitem__(self, index):
        item = self.X[index]

        if self.fn_aug is not None:
            item = self.fn_aug(item, self.args_aug)

        target = self.y[index]

        # mix_up
        import random
        if random.random() <= self.mix_up_rate:
            self.alpha = .2
            lam = np.random.beta(self.alpha, self.alpha, 1)
            lam = torch.from_numpy(max(lam, 1 - lam))
            idx = random.choice(self.class_ind[target.item()])
            item2 = self.X[idx]
            item = (lam * item + (1 - lam) * item2).float()

        item = [item, self.l_targets[index]]
        if self.return_index:
            item.append(item, index)

        return item, target

    def __len__(self):
        return len(self.X)

    def save(self, file_name):
        ch = {"X": self.X, "Y": self.y, "l": self.l, "stage": self.stage}
        torch.save(ch, file_name)

    @classmethod
    def init(cls, file_name,  aug_fn=None, args_aug=None,
             return_index=False, mix_up_rate=.2):
        ch = torch.load(file_name)
        X, l, y, stage = ch['X'], ch['l'], ch['Y'], ch['stage']
        return cls(X, l, y, stage, aug_fn, args_aug, return_index, mix_up_rate)


subjects = 15 # Num. of subjects used for LOSO
classes = 3 # Num of classes
num_sessions = 3
ds_path_frm = 'prev/processed2/{:s}_CV{:.0f}_{:.0f}.dataset'


def normalize(data):
    mee=np.mean(data,0)
    data=data-mee
    stdd=np.std(data,0)
    data=data/(stdd+1e-7)
    return data 

def get_data():
    # path = './emotion_data/SEED/ExtractedFeatures/'
    path = 'C:\\data\\SEED\\ExtractedFeatures\\'
    label = sio.loadmat(path+'label.mat')['label']
    files = sorted(glob.glob(path+'*_*'))

    sublist = set()
    for f in files:
        sublist.add(f.split('/')[-1].split('_')[0] )
    
    print('Total number of subjects: {:.0f}'.format(len(sublist)))
    sublist = sorted(list(sublist))
    print(sublist)

    sub_mov = [] 
    labels = []
    subject_label = []
    
    for sub_i in range(subjects):
        sub = sublist[sub_i]
        sub_files = glob.glob(sub+'_*')# sub_files = glob.glob(path+sub+'*')
        mov_data = [] 
        for f in sub_files:
            print(f)
            data = sio.loadmat(f, verify_compressed_data_integrity=False)
            keys = data.keys()
            de_mov = [k for k in keys if 'de_movingAve' in k] 
        
            mov_datai = [] 
            for t in range(15):
                temp_data = data[de_mov[t]].transpose(0,2,1)
                data_length  = temp_data.shape[-1]
                mov_i = np.zeros((62, 5, 265))
                mov_i[:,:,:data_length] = temp_data
                mov_i = mov_i.reshape(62, -1)#.transpose(1,0)
                mov_datai.append(mov_i)
            mov_datai = np.array(mov_datai)  
            mov_data.append(mov_datai) 
            
        mov_data = np.vstack(mov_data) 
        mov_data = normalize(mov_data) 
        sub_mov.append(mov_data)
        #labels.append(np.hstack([label, label, label]).squeeze())
        labels.append(np.hstack([label] * num_sessions).squeeze())
        n_trials = label.size
        subject_id = int(sub.split('\\')[-1])
        subj = np.array([subject_id]*n_trials*num_sessions)
        subject_label.append(subj)


    sub_mov = np.array(sub_mov) 
    labels = np.array(labels)
    subject_label = np.array(subject_label)

    return sub_mov, labels, subject_label


def get_SEED4():
    # path = './emotion_data/SEED/ExtractedFeatures/'
    path = 'C:\\data\\SEED\\ExtractedFeatures\\'
    label = sio.loadmat(path + 'label.mat')['label']
    files = sorted(glob.glob(path + '*_*'))

    sublist = set()
    for f in files:
        sublist.add(f.split('/')[-1].split('_')[0])

    print('Total number of subjects: {:.0f}'.format(len(sublist)))
    sublist = sorted(list(sublist))
    print(sublist)

    sub_mov = []
    labels = []
    subject_label = []

    for sub_i in range(subjects):
        sub = sublist[sub_i]
        sub_files = glob.glob(sub + '_*')  # sub_files = glob.glob(path+sub+'*')
        mov_data = []
        for f in sub_files:
            print(f)
            data = sio.loadmat(f, verify_compressed_data_integrity=False)
            keys = data.keys()
            de_mov = [k for k in keys if 'de_movingAve' in k]

            mov_datai = []
            for t in range(15):
                temp_data = data[de_mov[t]].transpose(0, 2, 1)
                data_length = temp_data.shape[-1]
                mov_i = np.zeros((62, 5, 265))
                mov_i[:, :, :data_length] = temp_data
                mov_i = mov_i.reshape(62, -1)  # .transpose(1,0)
                mov_datai.append(mov_i)
            mov_datai = np.array(mov_datai)
            mov_data.append(mov_datai)

        mov_data = np.vstack(mov_data)
        mov_data = normalize(mov_data)
        sub_mov.append(mov_data)
        # labels.append(np.hstack([label, label, label]).squeeze())
        labels.append(np.hstack([label] * num_sessions).squeeze())
        n_trials = label.size
        subject_id = int(sub.split('\\')[-1])
        subj = np.array([subject_id] * n_trials * num_sessions)
        subject_label.append(subj)

    sub_mov = np.array(sub_mov)
    labels = np.array(labels)
    subject_label = np.array(subject_label)

    return sub_mov, labels, subject_label


def split_signal(x, y, chunk_size=265):
    x_s, y_s = [], []
    x_n = []
    for k in x.keys():
        x_n.append(x[k])
    x_n = np.array(x_n).astype('float32')
    x_n = x_n.transpose((2,0,1))

    indices = np.where(np.diff(y) != 0)[0] + 1
    # append the length of the array to get the last part
    indices = np.append(indices, len(y))
    indices = np.insert(indices, 0, 0) # insert 0 at the first
    len_part = [indices[i] - indices[i - 1] for i in range(1, len(indices))]
    for lp, ind in zip(len_part, indices[:-1]):
        emotion = int(y[ind])
        x_p = np.zeros((62, 5, chunk_size))
        sz = min(chunk_size, lp)
        x_p[:,:,:sz] = x_n[:, :, ind:ind+sz]
        x_p = x_p.reshape(x_n.shape[0], -1)
        x_s.append(x_p)
        y_s.append(emotion)


    # split the array using the indices
    # parts = np.split(train_label, indices)
    # parts = np.split(x_train, indices, axis=0)

    # for lp, ind in zip(len_part, indices[:-1]):
    #     n_s = lp // chunk_size
    #     emotion = int(y[ind])
    #     for i in range(n_s):
    #         start = ind+i*chunk_size
    #         x_p = x_n[:, :, start:start + chunk_size].reshape(x_n.shape[0],-1)
    #         x_s.append(x_p)
    #         y_s.append(emotion)

    # x_s = np.array(x_n)
    # y_s = np.concatenate(y_s)
    return x_s, y_s


def get_SEED_GER(subjects=8, chunk_size=265):
    # chunk_size in terms of second
    # path = './emotion_data/SEED/ExtractedFeatures/'
    ds_path = 'C:/data/SEED/SEED_GER/DE/eeg_used_1s/'
    X, y, X_test, y_test = [], [], [], []
    l_train, l_test = [], []
    for sub_i in range(1, subjects+1):
        X_sub, y_sub, X_test_sub, y_test_sub = [], [], [], []
        files = sorted(glob.glob(ds_path + str(sub_i) + '_*'))
        for f in files:
            npz_data = np.load(f)
            # print(list(npz_data.keys()))  # train_data, test_data, train_label, test_label
            x_train = pickle.loads(npz_data['train_data'])
            x_test = pickle.loads(npz_data['test_data'])
            train_label, test_label = npz_data['train_label'], npz_data['test_label']
            x_s, y_s = split_signal(x_train, train_label, chunk_size)
            x_test_s, y_test_s = split_signal(x_test, test_label, chunk_size)
            X_sub += x_s
            X_test_sub += x_test_s
            y_sub += y_s
            y_test_sub += y_test_s

        X_sub = np.array(X_sub)
        # min_sig = X_sub.min(axis=0)
        # max_sig = X_sub.max(axis=0)
        # X_sub = (X_sub - min_sig) / (max_sig - min_sig)
        X_test_sub = np.array(X_test_sub)
        X_sub = np.vstack([X_sub, X_test_sub])
        X_sub = normalize(X_sub)

        X_test_sub = normalize(X_test_sub)

        y_sub = np.array(y_sub)
        l_sub = np.array([sub_i-1]*len(y_sub))
        y_test_sub = np.array(y_test_sub)
        l_test_sub = np.array([sub_i-1]*len(y_test_sub))

        y_sub = np.concatenate([y_sub, y_test_sub])
        l_sub = np.concatenate([l_sub, l_test_sub])

        X.append(X_sub), X_test.append(X_test_sub)
        y.append(y_sub), y_test.append(y_test_sub)
        l_train.append(l_sub), l_test.append(l_test_sub)

    X, X_test = np.vstack(X), np.vstack(X_test)
    y, y_test = np.concatenate(y), np.concatenate(y_test)
    l_train = np.concatenate(l_train)
    l_test = np.concatenate(l_test)
    from common.utils import conv2tensor
    X, y, l_train, X_test, y_test, l_test = conv2tensor(X, y, l_train, X_test, y_test, l_test)

    return X, y, l_train, X_test, y_test, l_test


def build_dataset(subjects):
    data_loaded = False
    for sub_i in range(subjects):
        train_path = ds_path_frm.format('train', subjects, sub_i)
        test_path = ds_path_frm.format('test', subjects, sub_i)
        print(train_path)
        
        if os.path.exists(train_path):
            print('Dataset already exists')
            continue

        if not data_loaded:
            mov_coefs, labels, l = get_data()
            data_loaded = True

        used_coefs = mov_coefs
        index_list = list(range(subjects))
        del index_list[sub_i]
        test_index = sub_i
        train_index = index_list

        print('Building train and test dataset')
        #get train & test
        X = used_coefs[train_index,:].reshape(-1, 62, 265*5)
        y_train = labels[train_index, :].reshape(-1)
        l_train = l[train_index].reshape(-1)
        X_test = used_coefs[test_index, :].reshape(-1, 62, 265 * 5)
        y_test = labels[test_index, :].reshape(-1)
        #get labels
        _, y_train = np.unique(y_train, return_inverse=True)
        _, y_test = np.unique(y_test, return_inverse=True)
        l_test = l[test_index]
        from utils import conv2tensor
        X, l_train, y = conv2tensor(X, l_train, y)
        train_dataset = AugDataset(X, l_train, y, 'train')
        train_dataset.save(train_path)
        X_test, l_test, y_test = conv2tensor(X_test, l_test, y_test)
        test_dataset = AugDataset(X_test, l_test, y_test, 'test')
        test_dataset.save(test_path)
        print('Dataset is built.')


def create_ds(ds:Datasets, subjects, chunk_size=265):
    if ds == Datasets.SEED_GER:
        X, y, l_train, X_test, y_test, l_test = get_SEED_GER(subjects, chunk_size=chunk_size)
        ds_path_ger = "ds/seed_ger.ds"
        save_dict={'X':X, 'y':y, 'l':l_train, 'X_test':X_test, 'y_test':y_test, 'l_test':l_test}
        torch.save(save_dict, ds_path_ger)
        print('dataset created successfully')
    if ds == Datasets.SEED:
        X, y, l = get_data()



def get_dataset(subjects, sub_i, mix_up_rate=0.):
    train_path = ds_path_frm.format('train', subjects, sub_i)
    test_path = ds_path_frm.format('test', subjects, sub_i)
    print(train_path)
    if not os.path.exists(train_path):
        raise IOError('Train dataset is not exist!')
    
    train_dataset = AugDataset.init(train_path, mix_up_rate=mix_up_rate)
    test_dataset = AugDataset.init(test_path, mix_up_rate=0.)

    return train_dataset, test_dataset


def gen_dl(ds:Datasets=Datasets.SEED, subjects=15, sub_i=0, bs=16, mixup_rate=0.):
    if ds == Datasets.SEED:
        # train_ds, test_ds = get_dataset(subjects, sub_i, mix_up_rate=mixup_rate)
        # train_loader = DataLoader(train_ds, batch_size=bs, drop_last=False, shuffle=True)
        # test_loader = DataLoader(test_ds, batch_size=bs, drop_last=False)
        cv_ds = CV_DS(ds=Datasets.SEED)
        train_loader, test_loader = cv_ds.gen_dl(sub_i)
    else:
        cv_ds = CV_DS()
        train_loader, test_loader = cv_ds.gen_dl(sub_i)

    return train_loader, test_loader


def gen_src_dl(ds=Datasets.SEED, bs=16):
    ds_path = ds_info[ds.name].ds_path
    ch = torch.load(ds_path)
    X, y, l = ch['X'], ch['y'].long(), ch['l'].long()
    s_dataset = AugDataset(X, l, y, 'train')
    s_loader = DataLoader(s_dataset, batch_size=bs, drop_last=False, shuffle=True)
    s_loader.ds = ds
    return s_loader


class CV_DS:
    def __init__(self, ds=Datasets.SEED_GER):
        self.n_subj = ds_info[ds.name].n_subj
        ds_path = ds_info[ds.name].ds_path
        ch = torch.load(ds_path)
        self.X, self.y, self.l = ch['X'], ch['y'].long(), ch['l'].long()
        self.ds = ds

    def load_cv_ds(self, sub_i=0):
        mask = self.l != sub_i
        X, y, l = self.X[mask], self.y[mask], self.l[mask]
        m_test = self.l == sub_i
        X_test, y_test, l_test = self.X[m_test], self.y[m_test], self.l[m_test]

        train_dataset = AugDataset(X, l, y, 'train')
        test_dataset = AugDataset(X_test, l_test, y_test, 'test')
        return train_dataset, test_dataset

    def gen_dl(self, sub_i=0, bs=16):
        train_ds, test_ds = self.load_cv_ds(sub_i)
        train_loader = DataLoader(train_ds, batch_size=bs, drop_last=False, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=bs, drop_last=False)
        return train_loader, test_loader


def deap_preprocess(data_file, dimension, T=128):
    # set the file type and path
    rnn_suffix = ".mat_win_%d_rnn_dataset.pkl" % (T)
    label_suffix = ".mat_win_%d_labels.pkl" % (T)
    arousal_or_valence = dimension
    with_or_without = 'yes'
    data_dir = "./deap_shuffled_data/"
    label_dir = "./deap_shuffled_data/" + with_or_without + "_" + arousal_or_valence + "/"

    with open(data_dir + data_file + rnn_suffix, "rb") as fp:
        rnn_datasets = pickle.load(fp)
    with open(label_dir + data_file + label_suffix, "rb") as fp:
        labels = pickle.load(fp)
        # labels = np.transpose(labels)
        # labels = np.asarray(pd.get_dummies(labels), dtype=np.long)

    # shuffle data
    index = np.array(range(0, len(labels)))
    np.random.shuffle(index)
    rnn_datasets = rnn_datasets[index]  # .transpose(0,2,1)
    labels = labels[index]
    datasets = rnn_datasets

    datasets = datasets.reshape(-1, 1, 32, T).astype('float32')
    labels = labels.astype('int64')
    return datasets, labels


def load_deap(subjects, params):
    # with open(root_data + '\\s01.dat', 'rb') as file:
    #     data = pickle.load(file, encoding='latin1')
    X = np.empty(shape=(0, 1, params.n_channel, params.window_size), dtype='float32')
    y, l = np.array([], dtype='int64'), np.array([], 'int64')

    for i, subject in enumerate(subjects):
        print('processing subject = %s' % subject)
        dim = params.dimension if hasattr(params, 'dimension') else params.dimention
        Xs, ys = deap_preprocess(subject, dim)
        ls = np.ones_like(ys) * (i + 1)
        X = np.concatenate((X, Xs), axis=0)
        y = np.concatenate((y, ys), axis=0)
        l = np.concatenate((l, ls), axis=0)

    return X, y, l


def gen_cd_dls(src_ds:Datasets, tar_ds:Datasets, subj_test=0, bs=16):
    s_loader = gen_src_dl(src_ds, bs=bs)
    cv_ds = CV_DS(ds=tar_ds)
    tar_loader, test_loader = cv_ds.gen_dl(sub_i=subj_test, bs=bs)
    return s_loader, tar_loader, test_loader


def main():
    # X, y, l = get_data()
    # create_ds(ds=Datasets.SEED_GER, subjects=8)
    # cv_ds = CV_DS(ds=Datasets.SEED)

    s_loader, tar_loader, test_loader = gen_cd_dls(src_ds=Datasets.SEED, tar_ds=Datasets.SEED4,
                                                   subj_test=0, bs=16)
    it = iter(tar_loader)
    batch = next(it)
    print(len(batch))

    it = iter(s_loader)
    batch = next(it)
    print(len(batch))

    # deap_cv_ds = CV_DS()
    # train_ds, test_ds = deap_cv_ds.load_cv_ds(32, 0, dim=Emotion_Dim.valence, use_majority_label=True)
    # vt = valence.transpose(1, 0)
    # temp = [torch.bincount(vt[i] + 1, minlength=3).view(1, -1) for i in range(vt.size(0))]
    # temp2 = torch.cat(temp)
    # major_ind = temp2.argmax(dim=1)
    # valence_major = torch.tile(major_ind.view(-1, 1), (1,32))
    # valence_major = valence_major.transpose(1,0)
    # train_dataset, test_dataset = deap_cv_ds.load_cv_ds(32, 0)
    pass


if __name__ == '__main__':
    main()