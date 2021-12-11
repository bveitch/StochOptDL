import numpy as np
import matplotlib.pyplot as plt
import argparse
import timeit

from scipy.optimize import minimize
from scipy.optimize import check_grad

from functools import partial
from nn_defn import get_initial_params_nL, forward_prop
from nn_objfn import NNObjfnBatch
from nn_solvers import SciPyNNWrap, GradSolver
from nn_utils import fill_vals, nn_test, compute_cost_accuracy, one_hot_labels, read_data
from vecWrap import VecWrap, DictToArray
# hyperparameters
HPARAMS = {
    'batch_size' : 1000,
    'num_epochs' : 100,
    'learning_rate' : 0.4,
    #'learning_rate' : 0.04,
    'num_hiddens' : [300, 100],
    #'reg' : 0.001
    'reg' : 0.01
    #'reg' : 0.1
}

def nn_train_nL(
    train_data, train_labels, dev_data, dev_labels,
    method = 'Gradient', reg=HPARAMS['reg'],
    num_hiddens=HPARAMS['num_hiddens'], learning_rate=HPARAMS['learning_rate'],
    num_epochs=HPARAMS['num_epochs'], batch_size=HPARAMS['batch_size']):

    print(f'Method            : {method}')
    print(f'Num hidden layers : {len(num_hiddens)}')
    print(f'Learning rate     : {learning_rate}')
    print(f'Num epochs        : {num_epochs}')
    print(f'Batch size        : {batch_size}')

    (nexp, dim) = train_data.shape

    init_params = get_initial_params_nL(dim, num_hiddens, 10)
    for p, w in init_params.items():
        print(f'{p} = {w.shape}')

    converter = DictToArray(init_params) 

    objfn = NNObjfnBatch(train_data, train_labels, reg, batch_size)
    cost_accuracy_train = partial(compute_cost_accuracy, train_data, train_labels)
    cost_accuracy_dev = partial(compute_cost_accuracy, dev_data, dev_labels)

    if method == 'Gradient':
        solver = GradSolver(num_epochs=num_epochs,objfn=objfn, learning_rate=learning_rate, \
        cost_accuracy_train=cost_accuracy_train, cost_accuracy_dev=cost_accuracy_dev)
    else:
        solver = SciPyNNWrap(num_epochs=num_epochs,objfn=objfn, learning_rate=learning_rate, \
            cost_accuracy_train=cost_accuracy_train, cost_accuracy_dev=cost_accuracy_dev,\
            method=method,converter=converter)

    start = timeit.default_timer()
    params = solver.solve(init_params)
    stop = timeit.default_timer()

    print('Total Time : ', stop- start, ' number of iterations', solver.nfeval)
    print('Time per iteration : ', (stop- start)/solver.nfeval)

    return params, fill_vals(solver.cost_train, num_epochs), fill_vals(solver.cost_dev, num_epochs), \
            fill_vals(solver.accuracy_train, num_epochs), fill_vals(solver.accuracy_dev, num_epochs)

def run_train_test(name,all_data, all_labels, num_epochs, plot=True, test_set = False):

    methods = ['Gradient','CG','L-BFGS-B']
    linestyles=dict(zip(methods,['-','--','-.']))
    markers=dict(zip(methods,[',',',', ',']))
    colors=dict(zip(methods,['r','b','c']))
    stats = {}
    fig, ax=plt.subplots(2,2)
    ax[0][0].set_ylabel('Loss')
    ax[1][0].set_ylabel('Accuracy')
    ax[1][0].set_xlabel('epochs')
    ax[1][1].set_xlabel('epochs')           
    #ax1.set_title('Costs: Train vs Dev')
    #ax2.set_title('Accuracy: Train vs Dev')

    for method in methods:
        params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train_nL(
            all_data['train'], all_labels['train'],
            all_data['dev'], all_labels['dev'],
            method=method,reg=HPARAMS['reg'],
            num_hiddens=HPARAMS['num_hiddens'], learning_rate=HPARAMS['learning_rate'],
            num_epochs=HPARAMS['num_epochs'], batch_size=HPARAMS['batch_size']
        )

        t = np.arange(num_epochs)

        if plot:
            if cost_train is not None:
                ax[0][0].semilogy(t, cost_train,color=colors[method], label='train_' + method, marker=markers[method], linestyle=linestyles[method])
                ax[0][0].legend()
            if cost_dev is not None:
                ax[0][1].semilogy(t, cost_dev,  color=colors[method], label='dev_' + method, marker=markers[method], linestyle=linestyles[method])
                ax[0][1].legend()
            if accuracy_train is not None:
                ax[1][0].plot(t, accuracy_train,color=colors[method], label='train_' + method, marker=markers[method], linestyle=linestyles[method])
                ax[1][0].legend()
            if accuracy_dev is not None: 
                ax[1][1].plot(t, accuracy_dev,  color=colors[method], label='dev_' + method, marker=markers[method] , linestyle=linestyles[method])
                ax[1][1].legend()

        if test_set:
            accuracy = nn_test(all_data['test'], all_labels['test'], params)
            print('For method %s, achieved test set accuracy: %f' % (method, accuracy))
    
    plt.savefig('./' + name + '.pdf')


def main(num_epochs=HPARAMS['num_epochs'], plot=True, train_baseline = True, train_regularized=True, test_set = False):
    np.random.seed(100)
    train_data, train_labels = read_data('./images_train.csv', './labels_train.csv')
    train_labels = one_hot_labels(train_labels)
    p = np.random.permutation(60000)
    train_data = train_data[p,:]
    train_labels = train_labels[p,:]

    dev_data = train_data[0:10000,:]
    dev_labels = train_labels[0:10000,:]
    train_data = train_data[10000:,:]
    train_labels = train_labels[10000:,:]

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    test_data, test_labels = read_data('./images_test.csv', './labels_test.csv')
    test_labels = one_hot_labels(test_labels)
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }
    nL=len(HPARAMS['num_hiddens'])
    run_train_test( 'compare_solvers_nL_%s_nepoch_%s_reg_%s_lr_%s' % (nL, num_epochs,HPARAMS['reg'],HPARAMS['learning_rate']) ,all_data, all_labels, num_epochs, plot, test_set = test_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=HPARAMS['num_epochs'])

    args = parser.parse_args()

    main(num_epochs = args.num_epochs)
