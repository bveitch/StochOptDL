#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 20:26:21 2020

@author: myconda
"""

import numpy as np
from Rosenbrock import Rosenbrock
from Solver0 import Solver, LineSearch 
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

x0=np.array([0,0])
xv=np.arange(-0.5,1.25,0.01)
yv=np.arange(-0.25,1.25,0.01)
zv = np.zeros((yv.size,xv.size))
objfn=Rosenbrock(1.0,50.0)
for j in range(yv.size):
    for i in range(xv.size):
        zv[j,i]=objfn.value([xv[i],yv[j]])

xvals={}
objs={}
niter=100
init_stepl=0.1
def solver(args):
    use_est=False
    if 'use_estimate' in args:
        use_est=args['use_estimate']
    linesearch=LineSearch(init_stepl,use_est)
    solver=Solver(linesearch,niter, use_hess=args['use_hess'],gtol=0)
    xsol=solver.solve(x0,objfn)
    [xvals,fvals]=solver.get_stats()
    return [xsol,xvals,fvals]


options={'Gradient':{'use_hess':False},'Steplength estimator':{'use_hess':True,'use_estimate':True},'Newton':{'use_hess':True,'use_estimate':False}}
#options={'stepEst':{'use_hess':True,'use_estimate':True}}
#options={'stepEst':{'use_hess':True,'use_estimate':True},'hess':{'use_hess':True,'use_estimate':False}}
colors=dict(zip(options.keys(),['orange','c','r']))
lines=dict(zip(options.keys(),['k-','b-','r-']))

for option,args in options.items():
    [xsol,xvs,fvs]=solver(args)
    xvals[option]=xvs
    objs[option]=fvs

fig, ax = plt.subplots()
cs = ax.contourf(xv, yv, zv,40)
ax.set_xlabel('x')
ax.set_ylabel('y')
cbar = fig.colorbar(cs)


# min_niter=niter
# for _,v in objs.items():
#     if len(v) < min_niter:
#         min_niter=len(v)

for name,xvs in xvals.items():
    col=colors[name]
    xp=[xv[0] for xv in xvs]
    yp=[xv[1] for xv in xvs]
    ax.plot(xp,yp,color=col,label=name)
    # for i in range(min_niter-1):
    #     xv0=xval[i]
    #     xv1=xval[i+1]
    #     #ax.arrow(xv0[0], xv0[1], xv1[0]-xv0[0], xv1[1]-xv0[1], width=0.0005,head_width=0.02, head_length=0.025, shape='full',fc=col, ec=col)
plt.legend(loc='upper left')
plt.savefig('Rosenbrock_iterations.png')   
plt.show()

fig, ax=plt.subplots(1,1,squeeze=False, figsize=(18,10))
iters=np.arange(0,niter)
ax[0][0].set_title("Convergence for Rosenbrock function",fontsize=10)
ax[0][0].set_xlabel('Iterations')
ax[0][0].set_ylabel('Error')
ax[0][0].set_xlim([-0.1,niter+0.1])
ax[0][0].set_ylim([0.0001, 1.01])
handles=[]
for name,obj in objs.items():
    ls=lines[name]
    if len(obj) < niter:
        obj0=[obj[-1]]*niter
        obj0[0:len(obj)]=np.array(obj)/obj[0]
    else:
        obj0=np.array(obj)/obj[0]
    ax[0][0].semilogy(iters, obj0, ls,label=name)
    #handles.append(line)
ax[0][0].legend()
plt.savefig('Rosenbrock_convergence.png')
plt.show()
      
