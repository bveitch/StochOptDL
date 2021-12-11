#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 20:26:21 2020

@author: myconda
"""

import math
import numpy as np
import numpy.linalg as npla
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

def wolfe1(fx,gx,fx1,stepl, c1=0.0001):
    return (fx1<=fx-c1*stepl*np.linalg.norm(gx)**2) 

class LineSearch:

    def __init__(self,init_stepl, use_estimate):
        self.init_stepl=init_stepl
        self.estimate = use_estimate
        self.maxiter = 20
        self.fac = 0.5

    def calculate_stepl(self, gx,Hx=None, eps=1.0e-10):
        stepl=self.init_stepl
        #print('Hx=',Hx)
        #print('est=',self.estimate)
        if Hx is not None:
            if self.estimate:
                numer=np.linalg.norm(gx)**2
                denom=np.dot(gx.T,np.dot(Hx,gx))
                stepl= numer/(denom+eps)
            else:
                gx=np.dot(npla.inv(Hx),gx)
                stepl=1.0
        return [stepl,gx]

    def search(self, x0,fx0,gx,fn, stepl):
        #[stepl,gx]=self.calculate_stepl(gx,Hx, eps=1.0e-10)
        i=0
        descent=False
        while(not descent and i<self.maxiter):
            x1  = x0 - stepl*gx
            fx1 = fn(x1)
            descent= wolfe1(fx0,gx,fx1,stepl,c1=0.0001)
            #descent = (fx1 < (fx+mu))
            stepl *= self.fac
            i += 1
        return [x1, fx1]

class Solver:

    def __init__(self,linesearch, niter, use_hess=True, tol=1.0e-12,gtol=1.0e-12):
        self.line_search=linesearch
        self.niter=niter 
        self.use_hess=use_hess
        self.tol=tol
        self.gtol=gtol
        self.fvals=[]
        self.xvals=[]

    def solve(self, x, fn, callback=None):
        x0=x
        fx_prev=None
        for n in range(self.niter):
            #print('x0=',x0)
            self.xvals.append(x0)
            fx0,gx0,Hx0=fn.val_grad_hess(x0)
            #print('fx=',fx0)
            #Hx0+=math.sqrt(n)*np.eye(Hx0.shape[0])
            self.fvals.append(fx0)
            #gx0=fn.gradient(x0)
            if fx_prev and (abs(fx0-fx_prev)<self.tol):
                print('convergence!')
                break
            nmgx=npla.norm(gx0)**2
            if nmgx<self.gtol:
                print('grad norm < gtol')
                break

            #Hx0=None            
            if not self.use_hess:
                Hx0=None
 
            [stepl,gx]=self.line_search.calculate_stepl(gx0,Hx0)
            [x0,_]=self.line_search.search(x0,fx0,gx,fn.value, stepl=stepl)
            if callback:
                callback(fx0,x0)

            fx_prev=fx0
        return x0 

    def get_stats(self):
        return [self.xvals, self.fvals]  


