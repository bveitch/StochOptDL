import numpy as np
import util
from logistic_reg import LogisticRegressionObjFn, SolverWrap
from Solver0 import Solver, LineSearch 

def stoch_LogReg_solver(train_path, valid_path, clf_name, clf, batch_size):
    """Problem: Logistic regression with General Solver.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    print('LR Solver (%s) : batchsize %d ' % (clf_name,batch_size))

    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # Train a logistic regression classifier
    clf.fit(x_train, y_train)
    train_objs=clf.objs
    
    # Plot decision boundary on top of validation set set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
   
    print('... LR Train num_iterations (%s): %d' % (clf_name,len(train_objs)))
    print('... LR Train Loss (%s): %.4f' % (clf_name,train_objs[-1]))
  
    #Use np.savetxt to save predictions on eval set to save_path
    p_eval = clf.predict(x_eval)
    yhat = p_eval > 0.5
    print('... LR Test Accuracy (%s): %.4f' % (clf_name,np.mean( (yhat == 1) == (y_eval == 1))))
    return train_objs 
        
class StochLogisticRegressionObjFn(LogisticRegressionObjFn):
    
    def __init__(self, step_size=0.01, max_iter=50, eps=1e-10, method='Newton-LS', batch_size=1):
        LogisticRegressionObjFn.__init__(self,step_size=step_size, max_iter=max_iter, eps=eps, method=method)
        self.batch_size = batch_size

    class stoch_lr_objfn:

        def __init__(self, x, y, batch_size):
            assert y.shape[0] == x.shape[0]
            assert batch_size > 0
            self.x = x
            self.y = y
            self.batch_size = batch_size
            self.nbatches, r = divmod(len(self.y), self.batch_size)
            if r > 0:
                self.nbatches+=1
            self.subfns=[]
            self.set_subfns(self.x,self.y)

        def set_subfns(self,x,y):
            self.subfns.clear()
            for i in range(self.nbatches):
                n0=i*self.batch_size
                n1=min(n0+self.batch_size,len(y))
                subx = x[n0:n1,:]
                suby = y[n0:n1]
                sub_objfn = LogisticRegressionObjFn.lr_objfn(subx,suby)
                self.subfns.append(sub_objfn)

        def reshuffle(self):
            inds=random.shuffle(list(range(len(self.y))))
            x_new=x[inds,:]
            y_new=y[inds]
            self.set_subfns(x_new,y_new)

        def value(self, x):
            val=0
            for sub_objfn in self.subfns:
                val+=sub_objfn.value(x)
            return val / self.nbatches

        def grad(self,x):
            g=np.zeros((x.shape))
            for sub_objfn in self.subfns:
                g+=sub_objfn.grad(x)
            return g / self.nbatches

        def hess(self,x):
            H=self.subfns[0].hess(x)
            for i in range(1,self.nbatches):
                H+=self.subfns[i].hess(x)
            return H / self.nbatches

        def val_grad_hess(self,theta):
            v=self.value(theta)
            g=self.grad(theta)
            H=self.hess(theta)
            return [v,g,H]

    def fit(self,x,y):
        objfn=self.stoch_lr_objfn(x, y, self.batch_size)
        print(x.shape[1])
        solver=SolverWrap(self.max_iter,objfn,self.eps,self.step_size,self.method)
        solver.check_grad(np.random.normal(0, 1, x.shape[1]))
        theta=np.zeros(x.shape[1])
        self.theta=solver.solve(theta)
        self.objs=solver.objs

if __name__ == '__main__':

    batch_sizes=[1, 5,10,20,50,100]
    sclf_train_objs={}
    for batch_size in batch_sizes:
        sclf_solvers={'Newton (noLS)':StochLogisticRegressionObjFn(batch_size = batch_size, method='Newton-noLS'),
                'Newton (LS)':StochLogisticRegressionObjFn(batch_size = batch_size, method='Newton-LS'),
                 'NLCG': StochLogisticRegressionObjFn(batch_size = batch_size, method='CG'),
                 'BFGS':StochLogisticRegressionObjFn(batch_size = batch_size, method='BFGS')}
        sclf_train_objs[batch_size]={}
        for sclf_name, sclf_solve in sclf_solvers.items():
            sclf_train_objs[batch_size][sclf_name]=stoch_LogReg_solver(train_path='ds1_train.csv',
                        valid_path='ds1_valid.csv',
                        clf_name=sclf_name, 
                        clf=sclf_solve,
                        batch_size=batch_size)
    
    util.multi_plot_convergence(sclf_train_objs, 'stoch_logreg_converge_test_newbatches.png')

    batch_size=10
    util.plot_convergence(sclf_train_objs[batch_size], 'stoch_logreg_converge_test_batch_size_%s.png' % batch_size)

        

    