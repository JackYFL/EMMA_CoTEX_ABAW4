import numpy as np
from sklearn.metrics import f1_score


def copy_state_dict(cur_state_dict, pre_state_dict, prefix='', load_name=None):
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None
    for k in cur_state_dict.keys():
        if load_name is not None:
            if load_name not in k:
                continue
        v = _get_params(k)
        try:
            if v is None:
                continue
            cur_state_dict[k].copy_(v)
        except:
            continue


def averaged_f1_score(input, target):
    N, label_size = input.shape
    f1s = []
    for i in range(label_size):
        target_i = target[:, i]
        input_i = input[:, i]
        mask = (target_i != -1)
        target_select = target_i[mask]
        input_select = input_i[mask]
        f1 = f1_score(input_select, target_select)
        f1s.append(f1)
    return np.mean(f1s), f1s


def accuracy(input, target):
    assert len(input.shape) == 1
    return sum(input == target)/input.shape[0]


def averaged_accuracy(x, y):
    assert len(x.shape) == 2
    N, C = x.shape
    accs = []
    for i in range(C):
        acc = accuracy(x[:, i], y[:, i])
        accs.append(acc)
    return np.mean(accs), accs


def CCC_score(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc


def VA_metric(x, y):
    mask = ~(y[:, 0] == -5)
    ynew = y[mask]
    items = [CCC_score(x[:, 0], ynew[:, 0]), CCC_score(x[:, 1], ynew[:, 1])]
    return items, np.mean(items)


def EXPR_metric(x, y):
    if not len(x.shape) == 1:
        if x.shape[1] == 1:
            x = x.reshape(-1)
        else:
            x = np.argmax(x, axis=-1)
    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)
    mask = (x != -1)  # don't include the -1 label
    x_select = x[mask]
    y_select = y[mask]
    f1 = f1_score(x_select, y_select, average='macro')
    acc = accuracy(x_select, y_select)
    return [f1, acc]


def new_EXPR_metric(x, y):
    if not len(x.shape) == 1:
        if x.shape[1] == 1:
            x = x.reshape(-1)
        else:
            x = np.argmax(x, axis=-1)
    if not len(y.shape) == 1:
        if y.shape[1] == 1:
            y = y.reshape(-1)
        else:
            y = np.argmax(y, axis=-1)
    mask = (y != -1)  # don't include the -1 label
    x_select = x[mask]
    y_select = y[mask].astype('int')
    N = x_select.shape[0]
    x_onehots = np.zeros([N, 8])
    x_onehots[np.arange(N), x_select] = 1
    y_onehots = np.zeros([N, 8])
    y_onehots[np.arange(N), y_select] = 1
    f1_av, f1s = averaged_f1_score(x_onehots, y_onehots)
    acc = accuracy(x_select, y_select)
    return [f1_av, acc]


def AU_metric(x, y, thresh=np.array([0.8, 0.8, 0.7, 0.5, 0.5, 0.5, 0.6, 0.8, 0.8, 0.8, 0.3, 0.7])):
    x[x > thresh] = 1
    x[x <= thresh] = 0
    f1_av, f1s = averaged_f1_score(x, y)
    x = x.reshape(-1)
    y = y.reshape(-1)
    acc_av = accuracy(x, y)
    return [f1_av, acc_av, f1s]


def all_metrics(x, y):
    VA, VA_labels = x[:, :2], y[:, :2]
    exp, exp_labels = x[:, 2:10], y[:, 2]
    AU, AU_labels = x[:, 10:], y[:, 3:]
    au_f1_av, au_acc_av, au_f1s = AU_metric(AU, AU_labels)
    VA_items, VA_av = VA_metric(VA, VA_labels)
    exp_f1, exp_acc = new_EXPR_metric(exp, exp_labels)
    total_indicator = au_f1_av + VA_av + exp_f1

    return total_indicator, au_f1_av, VA_av, exp_f1, exp_acc
