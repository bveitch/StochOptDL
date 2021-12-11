import matplotlib.pyplot as plt
import numpy as np


def add_intercept(x):
    """Add intercept to matrix x.

    Args:
        x: 2D NumPy array.

    Returns:
        New matrix same as x with 1's in the 0th column.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x


def load_dataset(csv_path, label_col='y', add_intercept=False):
    """Load dataset from a CSV file.

    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 't').
         add_intercept: Add an intercept entry to x-values.

    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)

    # Validate label_col argument
    allowed_label_cols = ('y', 't')
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'
                         .format(label_col, allowed_label_cols))

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels


def plot(x, y, theta, save_path, name, correction=1.0):
    """Plot dataset and fitted logistic regression parameters.

    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply, if any.
    """
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Plot decision boundary (found by solving for theta^T x = 0)
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta[0] / theta[2] + theta[1] / theta[2] * x1
           + np.log((2 - correction) / correction) / theta[2])
    plt.plot(x1, x2, c='red', linewidth=2)
    plt.xlim(x[:, -2].min()-.1, x[:, -2].max()+.1)
    plt.ylim(x[:, -1].min()-.1, x[:, -1].max()+.1)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Data with decsion boundary - {:}'.format(name))
    plt.savefig(save_path)

def plot_convergence(objective_vals, save_path):

    lines=dict(zip(objective_vals.keys(),['k-','b-','r-','c-']))
    fig, ax = plt.subplots()
    ax.set_xlabel('iteration')
    ax.set_ylabel('obj')
    ax.set_title('Logistic Regression convergence')
    niter=max([len(vals) for vals in objective_vals.values()])
    ax.set_xlim([-0.5, niter+0.5])
    for name,objs in objective_vals.items():
        ls=lines[name]
        if len(objs) < niter:
            #objs0=[objs[-1]/objs[0]]*niter
            #objs0[0:len(objs)]=np.array(objs)/objs[0]
            objs0=[objs[-1]]*niter
            objs0[0:len(objs)]=np.array(objs)
        else:
            #objs0=np.array(objs)/objs[0]
            objs0=np.array(objs)
        iters=np.arange(0,niter)
        ax.plot(iters,objs0,ls,label=name)

    ax.legend(loc='upper right')

    plt.savefig(save_path)
    plt.show()

def multi_plot_convergence(objective_vals, save_path):
    
    assert len(objective_vals) == 6
    fig, ax=plt.subplots(2,3,squeeze=False, figsize=(18,10))
    plot_num=0
    for nbatch, objs in objective_vals.items():
        lines=dict(zip(objs.keys(),['k-','b-','r-','c-']))
        i,j=divmod(plot_num,3)
        print(nbatch)
        print(i,j)
        ax[i][j].set_title("Nbatches={:}".format(nbatch),fontsize=20)
        if i==1:
            ax[i][j].set_xlabel('Iterations')
        ax[i][j].set_ylabel('Error')
        niter=max([len(vals) for vals in objs.values()])
        ax[i][j].set_xlim([-0.1,niter+0.1])
        for name,obj in objs.items():
            print('line name:',name)
            ls=lines[name]
            if len(obj) < niter:
                obj0=[obj[-1]]*niter
                obj0[0:len(obj)]=np.array(obj)
            else:
                obj0=np.array(obj)
            iters=np.arange(0,niter)
            ax[i][j].plot(iters, obj0, ls,label=name)
        #handles.append(line)
        ax[i][j].legend(loc='upper right')
        plot_num+=1

    plt.savefig(save_path)
    plt.show()
