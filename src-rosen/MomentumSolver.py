#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 20:26:21 2020

@author: myconda
"""

from Solver0 import Solver, LineSearch, wolfe1
import numpy as np
import numpy.linalg as npla

class MomentumLineSearch(LineSearch):

    def __init__(self,init_stepl, use_estimate,mom=0.9):
        LineSearch.__init__(self,init_stepl, use_estimate)
        self.mom = 0.9

    def search(self, x0, delta0,fx0,gx,fn, stepl):
        #[stepl,gx]=self.calculate_stepl(gx,Hx, eps=1.0e-10)
        i=0
        descent=False
        delta0*=self.mom
        while(not descent and i<self.maxiter):
            delta1  = delta0 + stepl*gx
            x1 = x0 - delta1
            fx1 = fn(x1)
            descent= wolfe1(fx0,gx,fx1,stepl,c1=0.0001)
            #descent = (fx1 < (fx+mu))
            stepl *= self.fac
            i += 1
        return [x1, delta1, fx1]

class MomentumSolver(Solver):

    #def __init__(self,linesearch, niter, use_hess=True, tol=1.0e-12,gtol=1.0e-12):
    #    Solver.__init__(self,line_search,niter, use_hess, tol, gtol)
       
    def solve(self, x, fn):
        x0=x
        delta0=0
        fx_prev=None
        
        #v_Hx=None
        for n in range(self.niter):
            #print('x0=',x0)
            self.xvals.append(x0)
            fx0,gx0,Hx0=fn.val_grad_hess(x0)

            print('fx=',fx0)
            self.fvals.append(fx0)
            if fx_prev and (abs(fx0-fx_prev)<self.tol):
                print('convergence!')
                break
            nmgx=npla.norm(gx0)**2
            if nmgx<self.gtol:
                print('grad norm < gtol')
                break
          
            if not self.use_hess:
                Hx0=None
 
            [stepl,gx]=self.line_search.calculate_stepl(gx0,Hx0)
            [x0,delta0,_]=self.line_search.search(x0,delta0,fx0,gx,fn.value, stepl=stepl)
            fx_prev=fx0
        return x0 



