import torch
import numpy as np
from sklearn.metrics import accuracy_score

def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.size(0)
    m = Y.size(1)
    x = torch.norm(X, dim=1)
    
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))
    
    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = X.matmul(Y)
    
    result = x + y - 2 * crossing_term
    result = result.relu()

    if not square:
        result = torch.sqrt(result)
    return result


def resampling_labels(labels_true, labels_hat, time_true, time_hat):
    if not torch.is_tensor(labels_true):
        labels_true = torch.from_numpy(labels_true)
    if not torch.is_tensor(labels_hat):
        labels_hat = torch.from_numpy(labels_hat)
    
    change = labels_true[1:] -labels_true[:-1]
    mask = change != 0
    change_points = np.abs(change[mask])
    mask = np.concatenate((np.array([False]), mask))
    time_change = time_true[mask]

    labels_true_down = np.zeros(time_hat.size, dtype=int)
    time_change_size = time_change.size
    time_change_size -= time_change_size % 2
    for i in range(0, time_change_size, 2):
        mask = np.logical_and(time_change[i] < time_hat, time_hat < time_change[i+1])
        labels_true_down[mask] = change_points[i]

    return labels_true_down


def accuracy(labels_true, labels_hat, classes=None):
    if not torch.is_tensor(labels_true):
        labels_true=torch.from_numpy(labels_true)
    if not torch.is_tensor(labels_hat):
        labels_hat=torch.from_numpy(labels_hat)

    if not labels_true.size(0) == labels_hat.size(0):
        raise ValueError(f"The label vectors should have the same size but got {labels_true.size(0)} and {labels_hat.size(0)}")
    if not labels_true.dim() == 1:
        raise ValueError(f"The vector of true labels should have dimension 1 but got {labels_true.dim()}")
    if not labels_hat.dim() == 1:
        raise ValueError(f"The vector of predicted labels should have dimension 1 but got {labels_hat.dim()}")
    
    f1_score_list = []
    if classes == None:
        classes = labels_true.unique(sorted=True)
    else:
        pass
    for c in classes:
        not_c = torch.tensor([x for x in classes if x!=c])
        tp = torch.logical_and(labels_hat == c, labels_true == c).sum()
        fp = torch.logical_and(labels_hat == c, torch.isin(labels_true, not_c)).sum()
        fn = torch.logical_and(torch.isin(labels_hat, not_c), labels_true == c).sum()
        score = tp / (tp + 0.5*(fp + fn))
        score = score.cpu().detach().numpy()
        score = float(score)
        f1_score_list.append(score)

    # accuracy
    mask = torch.isin(labels_true, torch.tensor(classes))
    accuracy = accuracy_score(labels_true[mask], labels_hat[mask])
    
    return f1_score_list, accuracy

def init_membership(score_s, time, nb_classes, time_shift=0, default=0, one_hot=False):
    u_init = np.zeros((time.size, nb_classes)) + default
    for i in range(len(score_s)-1, -1, -1): 
        T0 = score_s["start"].iloc[i] - time_shift
        T = score_s["end"].iloc[i] - time_shift
        phase = score_s["phase"].iloc[i]
        mask = np.logical_and(T0<=time, time<=T)
        if phase == "A":
            cond = np.array([True, False, False, False])
        elif phase == "B":
            cond = np.array([False, False, True, False])
        elif phase == "C":
            cond = np.array([False, False, False, True])
        elif phase == "D":
            cond = np.array([False, True, False, False])
        u_init[mask,:] = 0
        u_init[mask,cond] = 1
    if not one_hot:
        u_init = np.argmax(u_init, axis=1) +1
    return u_init

