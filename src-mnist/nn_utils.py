import numpy as np

from scipy.optimize import minimize
from scipy.optimize import check_grad

from functools import partial
from nn_defn import get_initial_params_1L, get_initial_params_nL, forward_prop
from nn_objfn import NNObjfnBatch
from vecWrap import VecWrap, DictToArray

def fill_vals(vals, num_vals):
    if len(vals)==0:
        return None
    elif len(vals) < num_vals:
        vals0=[vals[-1]]*num_vals
        vals0[0:len(vals)]=vals
        return vals0
    else:
        return vals

def nn_test(data, labels, params):
    output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_cost_accuracy(data, labels, params):
    output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output,labels)
    return cost, accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) ==
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def read_data(images_file, labels_file, max_rows=None):
    if max_rows is None:
        x = np.loadtxt(images_file, delimiter=',')
        y = np.loadtxt(labels_file, delimiter=',')
    else:
        x = np.loadtxt(images_file, delimiter=',', max_rows = max_rows)
        y = np.loadtxt(labels_file, delimiter=',', max_rows = max_rows)
    return x, y
