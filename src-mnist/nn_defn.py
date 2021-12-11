import numpy as np

def softmax(x):
    """
    Compute softmax function for a batch of input values.
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output. When implementing softmax, you should be careful
    to only sum over the second dimension.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistent when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """
    # *** START CODE HERE ***
    # n,K=x.shape
    # softmax=np.zeros((x.shape))
    # b=x.max(axis=1)
    # """ for i in range(K):
    #     a_i=x[:,i]-b
    #     # B_i=np.delete(x,i,1)-np.expand_dims(a_i,axis=-1)
    #     # E_i=np.exp(B_i)
    #     # softmax[:,i]=1.0/(1+np.sum(E_i,axis=1))
    #     softmax[:,i]=np.exp(a_i)/(np.sum(np.exp(x),axis=1))
    # return softmax """
    # for i in range(n):
    #     a_i=x[i,:]-b[i]
    #     softmax[i,:]=np.exp(a_i)/np.sum(np.exp(a_i))
    # return softmax
    
    x=x-np.max(x,axis=1)[:,np.newaxis]
    exp=np.exp(x)
    softmax=exp/np.sum(exp,axis=1)[:,np.newaxis]
    return softmax
    # *** END CODE HERE ***

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    # *** START CODE HERE ***
    #return 1./(1.+np.exp(-x))
    sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    return sig
    # *** END CODE HERE ***

def get_initial_params_1L(input_size, num_hidden, num_output):
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

    # *** START CODE HERE ***
    #print(input_size)
    #print(num_hidden)
    W1=np.random.normal(loc=0.0,scale=1.0,size=(input_size,num_hidden))
    #W1=np.random.standard_normal(size=(input_size,num_hidden))
    #print(W1.shape)
    b1=np.zeros((num_hidden))
    #W2=np.random.standard_normal(size=(num_hidden,num_output))
    W2=np.random.normal(loc=0.0,scale=1.0,size=(num_hidden,num_output))
    b2=np.zeros((num_output))
    return {'W1':W1,'b1':b1,'W2':W2,'b2':b2}
    # *** END CODE HERE ***

def get_initial_params_nL(input_size, num_hiddens, num_output):
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

    params={}
    Nlayers=len(num_hiddens)+1
    curr_size=input_size
    for i in range(Nlayers):
        if i == len(num_hiddens):
            next_size = num_output
        else:
            next_size = num_hiddens[i]
        W = np.random.normal(loc=0.0,scale=1.0,size=(curr_size,next_size)) 
        b = np.zeros((next_size))
        curr_size = next_size
        params['W'+str(i+1)] = W
        params['b'+str(i+1)] = b
   
    return params


def forward_prop_1L(data, labels, params):
    """
    Implement the forward layer given the data, labels, and params.

    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """
    # *** START CODE HERE ***
    n=labels.shape[0]
    # print(n)
    # print(labels.shape)
    # print(data.shape)
    W1=params['W1']
    b1=params['b1']
    W2=params['W2']
    b2=params['b2']
    #print(W1.shape)
    #print(b1.shape)
    Z1=np.dot(data,W1)+b1
    #print('z1shape',Z1.shape)
    A1=sigmoid(Z1)
    #print('W2shape',W2.shape)
    #print('b2shape',b2.shape)
    Z2=np.dot(A1,W2)+b2
    #print('z2shape',Z2.shape)
    output=softmax(Z2)
    #old:
    #loss=-(1.0/n)*np.sum(np.multiply(labels,np.log(output)))
    log=np.where(output>0, np.log(output), -1000)
    loss=-(1.0/n)*np.sum(np.multiply(labels,log))
    #print('loss=',loss)
    return A1,output,loss
    # *** END CODE HERE ***

def forward_prop_2L(data, labels, params):
    """
    Implement the forward layer given the data, labels, and params.

    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """
    # *** START CODE HERE ***
    n=labels.shape[0]

    W1=params['W1']
    b1=params['b1']
    W2=params['W2']
    b2=params['b2']
    W3=params['W3']
    b3=params['b3']
 
    Z1=np.dot(data,W1)+b1
    A1=sigmoid(Z1)
    Z2=np.dot(A1,W2)+b2
    A2=sigmoid(Z2)
    Z3=np.dot(A2,W3)+b3
    output=softmax(Z3)
    #old:
    #loss=-(1.0/n)*np.sum(np.multiply(labels,np.log(output)))
    log=np.where(output>0, np.log(output), -1000)
    loss=-(1.0/n)*np.sum(np.multiply(labels,log))
    #print('loss=',loss)
    return A1,A2,output,loss
    # *** END CODE HERE ***

def forward_prop_nL(data, labels, params, Nlayers):
    """
    Implement the forward layer given the data, labels, and params.

    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """
    # *** START CODE HERE ***
    n=labels.shape[0]
 
    A=data
    outputs=[]
    for i in range(0,Nlayers):
        W=params['W'+str(i+1)]
        b=params['b'+str(i+1)]
        Z=np.dot(A,W)+b
        if i < ( Nlayers-1 ):
            A=sigmoid(Z)
        else:
            A=softmax(Z)
        outputs.append(A)
    #old:
    #loss=-(1.0/n)*np.sum(np.multiply(labels,np.log(output)))
    log=np.where(A>0, np.log(A), -1000)
    loss=-(1.0/n)*np.sum(np.multiply(labels,log))
    #print('loss=',loss)
    return outputs,loss
    # *** END CODE HERE ***

def forward_prop(data, labels, params):
    Nlayers, odd= divmod(len(params),2)
    assert odd==0, 'len(params) must be even'

    if Nlayers==2:
        _,output,loss = forward_prop_1L(data, labels, params)
    elif Nlayers==3:
        _,_,output,loss = forward_prop_2L(data, labels, params)
    else:
        outputs,loss = forward_prop_nL(data, labels, params, Nlayers)
        output = outputs[-1]
    return output, loss


def backward_prop_1L(data, labels, params):
    """
    Implement the backward propagation gradient computation step for a neural network

    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.

        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    W2=params['W2']
    [A1,output,loss]=forward_prop_1L(data, labels, params)
    n=labels.shape[0]
    #print(n)
    gradZ2=-(1.0/n)*(labels-output)
    #print('b2.shape',params['b2'].shape)
    #print('gZ2.shape',gradZ2.shape)
    gradW2=np.dot(A1.T,gradZ2)
    assert params['W2'].shape == gradW2.shape,"W2 shape doesnt match"
    #gradb2=gradZ2[0,:]
    gradb2=np.sum(gradZ2,axis=0)
    #print('gradb2.shape',gradb2.shape)
    assert params['b2'].shape == gradb2.shape,"b2 shape doesnt match"
    gradA1=np.dot(gradZ2,W2.T)
    #assert A1.shape == gradA1.shape,"A1 shape doesnt match"
    gradsigma=np.multiply(A1,np.ones((A1.shape))-A1)#broadcast error??
    gradZ1=np.multiply(gradA1,gradsigma)
    #print('W1.shape',params['W1'].shape)
    #print('b1.shape',params['b1'].shape)
    gradW1=np.dot(data.T,gradZ1)
    gradb1=np.sum(gradZ1,axis=0)
    assert params['W1'].shape == gradW1.shape,"W1 shape doesnt match"
    assert params['b1'].shape == gradb1.shape,"b1 shape doesnt match"
    return {"W1":gradW1,"b1":gradb1,"W2":gradW2,"b2":gradb2}
    # *** END CODE HERE ***

def backward_prop_2L(data, labels, params):
    """
    Implement the backward propagation gradient computation step for a neural network

    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.

        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    W3=params['W3']
    W2=params['W2']
    [A1,A2,output,loss]=forward_prop_2L(data, labels, params)
    n=labels.shape[0]
    #print(n)
    gradZ3=-(1.0/n)*(labels-output)
    #print('b2.shape',params['b2'].shape)
    #print('gZ2.shape',gradZ2.shape)
    gradW3=np.dot(A2.T,gradZ3)
    assert params['W3'].shape == gradW3.shape,"W3 shape doesnt match"
    #gradb2=gradZ2[0,:]
    gradb3=np.sum(gradZ3,axis=0)
    #print('gradb2.shape',gradb2.shape)
    assert params['b3'].shape == gradb3.shape,"b3 shape doesnt match"
    gradA2=np.dot(gradZ3,W3.T)

    gradsigma=np.multiply(A2,np.ones((A2.shape))-A2)#broadcast error??
    gradZ2=np.multiply(gradA2,gradsigma)
    gradW2=np.dot(A1.T,gradZ2)
    assert params['W2'].shape == gradW2.shape,"W2 shape doesnt match"
    gradb2=np.sum(gradZ2,axis=0)
    assert params['b2'].shape == gradb2.shape,"b2 shape doesnt match"
    gradA1=np.dot(gradZ2,W2.T)

    gradsigma=np.multiply(A1,np.ones((A1.shape))-A1)#broadcast error??
    gradZ1=np.multiply(gradA1,gradsigma)
    gradW1=np.dot(data.T,gradZ1)
    gradb1=np.sum(gradZ1,axis=0)
    assert params['W1'].shape == gradW1.shape,"W1 shape doesnt match"
    assert params['b1'].shape == gradb1.shape,"b1 shape doesnt match"
    return {"W1":gradW1,"b1":gradb1,"W2":gradW2,"b2":gradb2, "W3":gradW3,"b3":gradb3}
    # *** END CODE HERE ***

def backward_prop_nL(data, labels, params, Nlayers):
    """
    Implement the backward propagation gradient computation step for a neural network

    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.

        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
  
    [outputs,loss]=forward_prop_nL(data, labels, params, Nlayers)
    n=labels.shape[0]
    gradA=None
    dparams={}
    for i in range(Nlayers,0,-1):
        Wstr='W'+str(i)
        bstr='b'+str(i)
        A = outputs[i-1]
        W = params[Wstr]
        if i==Nlayers:
            gradZ=-(1.0/n)*(labels-A)
        else:
            gradsigma=np.multiply(A,np.ones((A.shape))-A)#broadcast error??
            gradZ=np.multiply(gradA,gradsigma)

        gradA=np.dot(gradZ,W.T)
        if i>1:
            gradW = np.dot(outputs[i-2].T,gradZ)
        else:
            gradW = np.dot(data.T,gradZ)
        gradb=np.sum(gradZ,axis=0)
        dparams[Wstr] = gradW
        dparams[bstr] = gradb

    return dparams
    # *** END CODE HERE ***

def backward_prop(data, labels, params):
    Nlayers, odd= divmod(len(params),2)
    assert odd==0, 'len(params) must be even'

    if Nlayers==2:
        return backward_prop_1L(data, labels, params)
    elif Nlayers==3:
        return backward_prop_2L(data, labels, params)
    else:
        return backward_prop_nL(data, labels, params, Nlayers)

def backward_prop_regularized(data, labels, params, reg):
    """
    Implement the backward propagation gradient computation step for a neural network

    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above
        reg: The regularization strength (lambda)

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.

        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    grad=backward_prop(data, labels, params)
    gradW1=grad["W1"]
    gradb1=grad["b1"]
    gradW2=grad["W2"]
    gradb2=grad["b2"]
    W1=params["W1"]
    W2=params["W2"]
    return {"W1":gradW1+reg*W1,"b1":gradb1,"W2":gradW2+reg*W2,"b2":gradb2}    
    # *** END CODE HERE ***

def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, backward_prop_func):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.

    This code should update the parameters stored in params.
    It should not return anything

    Args:
        train_data: A numpy array containing the training data
        train_labels: A numpy array containing the training labels
        learning_rate: The learning rate
        batch_size: The amount of items to process in each batch
        params: A dict of parameter names to parameter values that should be updated.
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API

    Returns: This function returns nothing.
    """

    # *** START CODE HERE ***
    n=train_labels.shape[0]
    print('tr.shape',train_labels.shape)
    print('batch.size',batch_size)
    nBatch=int(n/batch_size)
    print('nBatch=',nBatch)
    for nb in range(nBatch):
        print('nbatch=',nb)
        bL=nb*batch_size
        bU=bL+batch_size
        batch_labels=train_labels[bL:bU,:]
        batch_data=train_data[bL:bU,:]
        grad=backward_prop_func(batch_data, batch_labels, params)
        for param in params:
            params[param]-=learning_rate*grad[param]
    # *** END CODE HERE ***
