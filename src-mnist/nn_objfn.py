import numpy as np
from vecWrap import VecWrap
from nn_defn import forward_prop, backward_prop, backward_prop_regularized
from scipy.optimize import check_grad

class NNObjfn():

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def value(self, vec):
        wts = vec.params
        _, loss = forward_prop(self.data, self.labels, wts)
        return loss

    def grad(self,vec):
        wts = vec.params
        dwts = backward_prop(self.data, self.labels, wts)
        return VecWrap(dwts)

    def check_grad(self, v0, dv, eps=0.001, tol=1.0e-3):
        vp1 = self.value(v0+eps*dv)
        vm1 = self.value(v0-eps*dv)
        gv  = self.grad(v0)
        err = (vp1 - vm1)/(2.0*eps) - dv.dot(gv)
        print('err', err)
        assert abs(err) < tol, 'Gradient test failed: error {:3.4f}'.format(err)
        return True

    
class NNObjfnReg(NNObjfn):

    def __init__(self,data, labels, reg=None):
        NNObjfn.__init__(self,data, labels)
        self.reg = reg

    def value(self, vec):
        loss = NNObjfn.value(self, vec)
        if self.reg:
            loss += 0.5 * self.reg * vec.norm_sq()
        return loss

    def grad(self, vec):
        dvec = NNObjfn.grad(self, vec)
        if self.reg:
            dvec += self.reg * vec
        return dvec

class NNObjfnBatch(NNObjfnReg):

    def __init__(self,data, labels, reg, batch_size):
        NNObjfnReg.__init__(self, data, labels, reg) 
        self.batch_size = batch_size
        self.total_size=labels.shape[0]
        self.nBatch, rem=divmod(self.total_size, self.batch_size)
        print('total size : ', self.total_size)
        print('nbatch     : ',self.nBatch)
        assert rem == 0, 'incomplete batch division %s' % rem
        self.batches=[]
        for ib in range(self.nBatch):
            bL=ib*self.batch_size
            bU=min(bL+self.batch_size,self.total_size)
            self.batches.append((bL,bU))

    def objfn_batch(self, batch):
        labels_sub = self.labels[batch[0]:batch[1],:]
        data_sub = self.data[batch[0]:batch[1],:]
        return NNObjfnReg(data_sub, labels_sub, self.reg)
        
    def value(self, vec):
        loss=0
        for batch in self.batches:
            loss += self.objfn_batch(batch).value(vec)
        return loss * (1.0 / self.nBatch)

    def grad(self, vec):
        g = self.objfn_batch(self.batches[0]).grad(vec)
        for i in range(1,self.nBatch):
            g += self.objfn_batch(self.batches[i]).grad(vec)
        return g * (1.0 / self.nBatch)









