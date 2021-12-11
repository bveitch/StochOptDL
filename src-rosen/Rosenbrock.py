#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 20:26:21 2020

@author: myconda
"""

import numpy as np
import copy

class Rosenbrock:  
    
    def __init__(self, a, b):
        self.a = a
        self.b = b
    
    def value(self,v):
        x=v[0]
        y=v[1] 
        return (self.a - x)**2 + self.b*(y - x**2)**2
    
    def gradient(self,v):
        x=v[0]
        y=v[1] 
        gx = -2*(self.a - x) - 4*x*self.b*(y - x**2)
        gy =                   2  *self.b*(y - x**2)
        return np.array([gx,gy])
        
    def hessian(self,v):
        x=v[0]
        y=v[1] 
        hxx =  2 - 4*self.b*(y - x**2) + 8*x**2*self.b
        hxy = - 4*self.b*x
        hyy =   2*self.b
        return np.array([[hxx , hxy], [hxy , hyy] ])

    def val_grad_hess(self,v):
        return self.value(v), self.gradient(v), self.hessian(v)

class NoisyFunc:

    def __init__(self, mu, func):
        self.mu=mu
        self.f = func

    def val_grad_hess(self,v):
        v_=copy.deepcopy(v)
        if self.mu>0:
            noise=np.random.normal(0, self.mu, v.shape)
            v_+=noise
        return self.f.val_grad_hess(v_)

    def value(self,v):
        v_=copy.deepcopy(v)
        if self.mu>0:
            noise=np.random.normal(0, self.mu, v.shape)
            v_+=noise
        return self.f.value(v_)

class Average:

    def __init__(self,n_ave, func):
        self.n=n_ave
        self.func=func

    def val_grad_hess(self,v):
        val_=0
        grad_=0
        hess_=0
        for i in range(self.n):
            [val,grad,hess]=self.func.val_grad_hess(v)
            val_+=val
            grad_+=grad
            hess_+=hess
        return val_/self.n, grad_/self.n, hess_/self.n

    def value(self,v):
        val_=0
        for i in range(self.n):
            val=self.func.value(v)
            val_+=val
        return val_/self.n

class History:

    def __init__(self,nhistory):
        self.n=nhistory
        self.history=[]

    def add_v(self,val):
        if len(self.history)==0:
            self.history=[val]*self.n
        else:
            self.history.pop(0)
            self.history.append(val)

    def reset(self,history):
        assert len(history)==self.n, 'history to reset must have same size = {:}'.forrmat(self.n)
        self.history=history

    def get(self):
        print(self.history)
        print('sum=',sum(self.history))
        return 1.0*sum(self.history)/self.n


class HistoryFunc:

    def __init__(self,nhistory, func):
        self.n=nhistory
        self.history_val=History(nhistory)
        self.history_grad=History(nhistory)
        self.history_hess=History(nhistory)
        self.func=func

    def val_grad_hess(self,v):
        [val,grad,hess]=self.func.val_grad_hess(v)
        self.history_val.add_v(val)
        self.history_grad.add_v(grad)
        self.history_hess.add_v(hess)
        return self.history_val.get(), self.history_grad.get(), self.history_hess.get()

    def value(self,v):
        history_v=copy.deepcopy(self.history_val)
        val=self.func.value(v)
        history_v.add_v(val)
        return history_v.get()

class MomentumFunc:

    def __init__(self,delta, func):
        self.d=delta
        self.val=0
        self.grad=0
        self.hess=0
        self.func=func

    def val_grad_hess(self,v):
        [val,grad,hess]=self.func.val_grad_hess(v)
        self.val=self.d*self.val+val
        self.grad=self.d*self.grad+grad
        self.hess=self.d*self.hess+hess
        return self.val, self.grad, self.hess

    def value(self,v):
        tmp_val=copy.deepcopy(self.val)
        val=self.func.value(v)
        tmp_val=self.d*tmp_val+val
        return tmp_val   

       



    

     

