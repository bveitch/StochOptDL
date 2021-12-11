import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import check_grad
import util
from Solver0 import Solver, LineSearch 

def main_LogReg(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # Plot decision boundary on top of validation set set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_eval, y_eval, clf.theta, plot_path)

    # Use np.savetxt to save predictions on eval set to save_path
    p_eval = clf.predict(x_eval)
    yhat = p_eval > 0.5
    print('LR Accuracy: %.4f' % np.mean( (yhat == 1) == (y_eval == 1)))
    np.savetxt(save_path, p_eval)

def LogReg_solver(train_path, valid_path, save_path, clf_name, clf):
    """Problem: Logistic regression with General Solver.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    print('LR Solver (%s)' % (clf_name))
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # Train a logistic regression classifier
    clf.fit(x_train, y_train)
    train_objs=clf.objs
    
    # Plot decision boundary on top of validation set set
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_eval, y_eval, clf.theta,  plot_path, clf_name)
   
    print('... LR Train num_iterations (%s): %d' % (clf_name,len(train_objs)))
    print('... LR Train Loss (%s): %.4f' % (clf_name,train_objs[-1]))
  
    #Use np.savetxt to save predictions on eval set to save_path
    p_eval = clf.predict(x_eval)
    yhat = p_eval > 0.5
    print('... LR Test Accuracy (%s): %.4f' % (clf_name,np.mean( (yhat == 1) == (y_eval == 1))))
    np.savetxt(save_path, p_eval)
    return train_objs

def sigma(x):
    sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    return sig

def sigma0(x):
    return 1.0/(1 + np.exp(-x))


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=50, eps=1e-10,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        self.objs=[]

    # class lr_objfn:

    #     def __init__(self,x,y):
    #         self.x = x
    #         self.y = y
    #         self.n = len(y)

    #     def value(self, theta):
    #         z=sigma(self.y*np.dot(self.x,theta))
    #         return (-1.0/self.n)*np.sum(np.log(z))

    #     def grad(self,theta):
    #         r=sigma(np.dot(self.x,theta))-self.y
    #         return (1.0/self.n)*np.dot(self.x.T,r)

    #     def hess(self,theta):
    #         p=sigma(np.dot(self.x,theta))
    #         L= (1.0/self.n)*np.diag(p*(1.0-p))
    #         return np.dot(self.x.T,np.dot(L,self.x))

    # def fit_objfn_test(self,x,y):
    #     objfn=self.lr_objfn(x,y)
    #     theta=np.zeros(x.shape[1])
    #     delta=float('inf')
    #     i=0
    #     while delta>self.eps:
    #         objv=objfn.value(theta)
    #         print(i, objv)
    #         self.objs.append(objv)
    #         g=objfn.grad(theta)  
    #         H=objfn.hess(theta)
    #         dtheta=self.step_size*np.dot(np.linalg.inv(H),g)
    #         theta-=dtheta
    #         delta=sum(abs(dtheta)) 
    #         i+=1
    #     self.theta=theta

    # def fit_solver(self,x,y,args):
    #     lrobjfn=lr_objfn(x,y)
    #     linesearch=LineSearch(init_stepl,use_est)
    #     solver=Solver(linesearch,niter,args)
    #     theta0=np.zeros(x.shape[1])
    #     self.theta=solver.solve(theta0,lrobjfn)

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        xT=x.transpose()
        theta=np.zeros(x.shape[1])
        dtheta=theta
        delta=float('inf')
        it=0
        while it< self.max_iter and delta>self.eps:
            theta-=dtheta
            pred=1.0/(1.0+np.exp(-np.dot(x,theta))) 
            z=sigma(y*np.dot(x,theta))
            obj=(-1.0/len(y))*np.sum(np.log(z))
            self.objs.append(obj)
            L= np.diag(pred*(1.0-pred))
            res=y-pred  
            H=np.dot(xT,np.dot(L,x))
            b=np.dot(xT,-res)
            dtheta=np.dot(np.linalg.inv(H),b)
            delta=np.linalg.norm(dtheta,1)
            it+=1
        self.theta=theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1.0/(1+np.exp(-np.dot(x,self.theta)))
        # *** END CODE HERE ***

class SciPyWrap:

    def __init__(self, max_iter,objfn, eps, step_size, method):
        self.max_iter=max_iter
        self.objfn=objfn
        self.eps=eps
        self.method=method
        self.step_size=step_size
        self.objs=[]
        self.nfeval=0
        self.verbose=False

    def callbackFn(self,x):      
        self.nfeval+=1
        v=self.objfn.value(x)
        self.objs.append(v)
        if self.verbose:
            print('{0:4d} : {1:3.4f}'.format(self.nfeval,v))
 
    def solve(self,theta):
        x0=theta
        res=minimize(self.objfn.value,x0,method=self.method,jac=self.objfn.grad, 
        callback=self.callbackFn, tol=self.eps,options={'gtol':1.0e-12, 'maxiter':self.max_iter})
        return res['x']

    def check_grad(self,x,tol=1.0e-5):
        v = check_grad(self.objfn.value, self.objfn.grad, x)
        if v < tol:
            return True
        else:
            print('Gradient test failed: error {:3.4f}'.format(v))
            return False

class SolverWrap(SciPyWrap):
 
    def solve(self,theta):
        if self.method=='Newton-LS':
            linesearch=LineSearch(self.step_size,False)
            solver=Solver(linesearch, self.max_iter, use_hess=True, tol=self.eps,gtol=1.0e-12)
            res=solver.solve(theta, self.objfn)
            self.objs=solver.fvals
            self.nfeval=len(self.objs)
            return res
        elif self.method=='Newton-noLS':
            i=0
            delta=float('inf')
            while i< self.max_iter and delta>self.eps:
                #print('theta = ',theta)
                objv=self.objfn.value(theta)
                #print(i, objv)
                self.objs.append(objv)
                g=self.objfn.grad(theta)  
                H=self.objfn.hess(theta)
                dtheta=self.step_size*np.dot(np.linalg.inv(H),g)
                theta-=dtheta
                delta=sum(abs(dtheta)) 
                i+=1
            return theta
        else:
            return SciPyWrap.solve(self,theta)

class LogisticRegressionObjFn(LogisticRegression):

    class lr_objfn:

        def __init__(self,x,y):
            assert x.shape[0] == y.shape[0]
            self.x = x
            self.y = y
            self.n = len(y)

        def value(self, theta):
            w=np.dot(self.x,theta)
            z=sigma0(w)
            g=np.log(z)-(1-self.y)*w
            #g=self.y*np.log(z)+(1-self.y)*np.log(1-z)
            return (-1.0/self.n)*np.sum(g)

        def grad(self,theta):
            r=sigma(np.dot(self.x,theta))-self.y
            return (1.0/self.n)*np.dot(self.x.T,r)

        def hess(self,theta):
            p=sigma(np.dot(self.x,theta))
            L= (1.0/self.n)*np.diag(p*(1.0-p))
            return np.dot(self.x.T,np.dot(L,self.x))

        def val_grad_hess(self,theta):
            v=self.value(theta)
            g=self.grad(theta)
            H=self.hess(theta)
            return [v,g,H]

    # def fit(self,x,y):
    #     objfn=self.lr_objfn(x,y)
    #     theta=np.zeros(x.shape[1])
    #     delta=float('inf')
    #     i=0
    #     while i< self.max_iter and delta>self.eps:
    #         #print('theta = ',theta)
    #         objv=objfn.value(theta)
    #         #print(i, objv)
    #         self.objs.append(objv)
    #         g=objfn.grad(theta)  
    #         H=objfn.hess(theta)
    #         dtheta=self.step_size*np.dot(np.linalg.inv(H),g)
    #         theta-=dtheta
    #         delta=sum(abs(dtheta)) 
    #         i+=1
    #     self.theta=theta
      
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-10,method='Newton-LS'):
        LogisticRegression.__init__(self,step_size=step_size, max_iter=max_iter, eps=eps)
        self.method=method

    def fit(self, x, y):
        if self.method == 'XCS229i':
            LogisticRegression.fit(x,y)
        else:
            objfn=self.lr_objfn(x,y)
            solver=SolverWrap(self.max_iter,objfn,self.eps,self.step_size,self.method)
            solver.check_grad(np.random.normal(0, 1, x.shape[1]))
            theta=np.zeros(x.shape[1])
            self.theta=solver.solve(theta)
            self.objs=solver.objs

if __name__ == '__main__':
    # main_LogReg(train_path='ds1_train.csv',
    #      valid_path='ds1_valid.csv',
    #      save_path='logreg_pred_1.txt')

    # main_LogReg(train_path='ds2_train.csv',
    #      valid_path='ds2_valid.csv',
    #      save_path='logreg_pred_2.txt')

    clf_solvers={'Newton (noLS)':LogisticRegressionObjFn(),
                 'Newton (LS)':LogisticRegressionSciPy('Newton-LS'),
                 'NLCG': LogisticRegressionSciPy('CG'),
                 'L-BFGS':LogisticRegressionSciPy('L-BFGS')}

    clf_train_objs={}
    for clf_name, clf_solve in clf_solvers.items():
        save_path='logReg_pred_{:}.txt'.format(clf_name)
        clf_train_objs[clf_name]=LogReg_solver(train_path='ds1_train.csv',
                      valid_path='ds1_valid.csv',
                      save_path=save_path, 
                      clf_name=clf_name, 
                      clf=clf_solve)
    
    util.plot_convergence(clf_train_objs, 'logreg_converge_test_2.png')

    