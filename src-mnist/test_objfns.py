import numpy as np
from nn_objfn import NNObjfn, NNObjfnReg, NNObjfnBatch
from run_train_test import one_hot_labels, read_data, SciPyNNWrap
from vecWrap import VecWrap, DictToArray

HPARAMS_1L = {
    'batch_size' : 1000,
    'num_epochs' : 30,
    'learning_rate' : 0.4,
    'num_hiddens' : [300],
    #'reg' : 0.001
    'reg' : 0.001
}
HPARAMS_2L = {
    'batch_size' : 1000,
    'num_epochs' : 30,
    'learning_rate' : 0.4,
    'num_hiddens' : [300,100],
    #'reg' : 0.001
    'reg' : 0.001
}

HPARAMS_3L = {
    'batch_size' : 1000,
    'num_epochs' : 30,
    'learning_rate' : 0.4,
    'num_hiddens' : [300,100, 50],
    #'reg' : 0.001
    'reg' : 0.001
}
def get_random_params_1L(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer of size input_size x num_hidden
    b1 is the bias vector for the hidden layer of size num_hidden
    W2 is the weight matrix for the output layers of size num_hidden x num_output
    b2 is the bias vector for the output layer of size num_output

    As specified in the PDF, weight matrices should be initialized with a random normal distribution
    centered on zero and with scale 1.
    Bias vectors should be initialized with zero.

    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes

    Returns:
        A dict mapping parameter names to numpy arrays
    """

    W1=np.random.normal(loc=0.0,scale=1.0,size=(input_size,num_hidden))
    b1=np.random.normal(loc=0.0,scale=1.0,size=num_hidden)
    W2=np.random.normal(loc=0.0,scale=1.0,size=(num_hidden,num_output))
    b2=np.random.normal(loc=0.0,scale=1.0,size=num_output)
    return {'W1':W1,'b1':b1,'W2':W2,'b2':b2}

def get_random_params_nL(input_size, num_hiddens, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    As specified in the PDF, weight matrices should be initialized with a random normal distribution
    centered on zero and with scale 1.
    Bias vectors should be initialized with zero.

    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes

    Returns:
        A dict mapping parameter names to numpy arrays
    """

    params={}
    Nlayers=len(num_hiddens)+1
    curr_size=input_size
    for i in range(Nlayers):
        if i == len(num_hiddens):
            next_size = num_output
        else:
            next_size = num_hiddens[i]
        W = np.random.normal(loc=0.0,scale=1.0,size=(curr_size,next_size)) 
        b = np.random.normal(loc=0.0,scale=1.0,size=next_size)
        curr_size = next_size
        params['W'+str(i+1)] = W
        params['b'+str(i+1)] = b
   
    return params

def get_objfn(train_data, train_labels,reg=None,batch_size=None):

    if reg: 
        if batch_size :
            return NNObjfnBatch(train_data, train_labels, reg, batch_size)
        else:
            return NNObjfnReg(train_data, train_labels, reg)
    return NNObjfn(train_data, train_labels)

def test_objfn(train_data, train_labels, params):

    (nexp, dim) = train_data.shape
    reg         = params['reg']
    batch_size  = params['batch_size']
    num_hiddens = params['num_hiddens']
    
    print(f'Num hidden:    {num_hiddens}')
    print(f'Reg:           {reg}')
    print(f'Batch size:    {batch_size}')
    
    if len(num_hiddens)==1:
        wts = get_random_params_1L(dim, num_hiddens[0], 10)
        dwts = get_random_params_1L(dim, num_hiddens[0], 10)
    else:
        wts = get_random_params_nL(dim, num_hiddens, 10)
        dwts = get_random_params_nL(dim, num_hiddens, 10)
    
    objfn = get_objfn(train_data, train_labels,reg=reg,batch_size=batch_size)

    vec = VecWrap(wts)
    dvec = VecWrap(dwts)    
    return objfn.check_grad(vec, dvec, eps=0.001)

def test_objfn_scipy(
    train_data, train_labels,
    params):

    (nexp, dim) = train_data.shape

    params = get_random_params(dim, num_hidden, 10)

    converter = DictToArray(params) 

    print(f'Num hidden:    {num_hidden}')
    print(f'Reg:    {reg}')
    print(f'Batch size:     {batch_size}')
    
    objfn = get_objfn(train_data, train_labels,
    num_hidden=num_hidden,reg=reg,batch_size=batch_size)
    
    solver = SciPyNNWrap(num_epochs=0,objfn=objfn, learning_rate=0.0, \
            cost_accuracy_train=None, cost_accuracy_dev=None,\
            method=None,converter=converter)

    return solver.check_grad(params)

def run_test(test_params, data_sizes):
    np.random.seed(100)
    passed = True
    for data_size in data_sizes:
        train_data, train_labels = read_data('./images_train.csv', './labels_train.csv', data_size)
        
        print('train_data. size = ', train_data.shape)
        print('train_labels. size = ', train_labels.shape)

        train_labels = one_hot_labels(train_labels)
        p = np.random.permutation(data_size)
        train_data = train_data[p,:]
        train_labels = train_labels[p,:]

        train_data = train_data[0:data_size,:]
        train_labels = train_labels[0:data_size,:]

        mean = np.mean(train_data)
        std = np.std(train_data)
        train_data = (train_data - mean) / std

        for test_name, test_param in test_params.items():
            print('Testing : ', test_name)
            passed *= test_objfn(train_data, train_labels, test_param)
    return passed

def get_tests(hyper_params):
    return {'test_no_reg':{'reg':None, 'batch_size':None, 'num_hiddens':hyper_params['num_hiddens']},\
            'test_reg':{'reg':hyper_params['reg'], 'batch_size':None, 'num_hiddens':hyper_params['num_hiddens']},\
            'test_batch':{'reg':hyper_params['reg'], 'batch_size':hyper_params['batch_size'], 'num_hiddens':hyper_params['num_hiddens']}}

if __name__ == '__main__':
    test_params = get_tests(HPARAMS_3L)
    data_sizes =[ 1000, 10000 ]
    passed = run_test(test_params, data_sizes)
    if passed:
        print('All tests passed')


