import numpy as np

from scipy.optimize import minimize
from scipy.optimize import check_grad

from functools import partial
from nn_defn import get_initial_params_1L, get_initial_params_nL, forward_prop
from nn_objfn import NNObjfnBatch
from vecWrap import VecWrap, DictToArray


class GradSolver:

    def __init__(self, num_epochs,objfn, learning_rate, cost_accuracy_train, cost_accuracy_dev):
        self.num_epochs=num_epochs
        self.objfn=objfn
        self.learning_rate=learning_rate
        self.cost_accuracy_train = cost_accuracy_train
        self.cost_accuracy_dev = cost_accuracy_dev
        self.cost_train = []
        self.cost_dev = []
        self.accuracy_train = []
        self.accuracy_dev = []
        self.nfeval=0
        self.verbose=True

    def callbackFn(self,vec):      
        if self.verbose and self.nfeval==0:
            print('Iter : Trn cost : Trn acc : Dev cost : Dev acc ')
        wts = vec.params
        tr_cost, tr_accuracy = self.cost_accuracy_train(wts)
        self.cost_train.append(tr_cost)
        self.accuracy_train.append(tr_accuracy)
        cost, accuracy = self.cost_accuracy_dev(wts)
        self.cost_dev.append(cost)
        self.accuracy_dev.append(accuracy)
        if self.verbose:
            print('{0:4d} : {1:2.4f} : {2:2.4f} : {3:2.4f} : {4:2.4f}'.format(self.nfeval,tr_cost, tr_accuracy, cost, accuracy))
        self.nfeval+=1
 
    def solve(self,wts):
        vec = VecWrap(wts)
        for epoch in range(self.num_epochs):
            gvec = self.objfn.grad(vec)
            self.callbackFn(vec)
            vec -= self.learning_rate * gvec
        return vec.params

class SciPyNNWrap(GradSolver):

    def __init__(self, num_epochs,objfn, learning_rate, cost_accuracy_train, cost_accuracy_dev, method, converter):
        GradSolver.__init__(self, num_epochs,objfn, learning_rate, cost_accuracy_train, cost_accuracy_dev)
        self.eps=1.0e-12
        self.method=method
        self.converter = converter

    def convert_fwd(self,vec):
        return self.converter.forward(vec.params)

    def convert_bwd(self,x):
        return VecWrap(self.converter.backward(x))

    def callbackFn(self,x):      
        vec=self.convert_bwd(x)
        GradSolver.callbackFn(self,vec)

    def loss_fn(self, x):
        vec=self.convert_bwd(x)
        return self.objfn.value(vec)

    def loss_jac(self,x):
        vec=self.convert_bwd(x)
        gvec = self.objfn.grad(vec)
        return self.convert_fwd(gvec)
    
    def solve(self,wts):
        vec = VecWrap(wts)
        x0=self.convert_fwd(vec)
        res=minimize(self.loss_fn,x0,method=self.method,jac=self.loss_jac, \
            callback=self.callbackFn, tol=self.eps,options={'gtol':1.0e-12, 'maxiter':self.num_epochs})
        vec_soln = self.convert_bwd(res['x'])
        return vec_soln.params

    def check_grad(self,wts,tol=1.0e-5):
        vec = VecWrap(wts)
        x0=self.convert_fwd(vec)
        v = check_grad(self.loss_fn, self.loss_jac, x0)
        assert v < tol, 'Gradient test failed: error {:3.4f}'.format(v)
        return True
