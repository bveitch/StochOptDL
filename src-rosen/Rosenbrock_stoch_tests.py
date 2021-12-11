#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 20:26:21 2020

@author: myconda
"""

import numpy as np
from Rosenbrock import Rosenbrock, NoisyFunc, MomentumFunc, Average
from Solver0 import Solver, LineSearch 
from MomentumSolver import MomentumSolver, MomentumLineSearch
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

noise_mu=0.01
noisy_objfn=NoisyFunc(noise_mu,objfn)

xvals={}
objs={}
niter=200
init_stepl=0.1

def solver(args,nave):
    use_est=args.get('use_estimate',False)
    #linesearch=LineSearch(init_stepl,use_est)
    #solver=MomentumSolver(linesearch,niter, use_hess=args['use_hess'],gtol=0)
    use_mom=args.get('momentum',False)
    if use_mom:
        linesearch=MomentumLineSearch(init_stepl,use_est)
        solver=MomentumSolver(linesearch,niter, use_hess=args['use_hess'],gtol=0)
    else:
        linesearch=LineSearch(init_stepl,use_est)
        solver=Solver(linesearch,niter, use_hess=args['use_hess'],gtol=0)
    stoch_objfn=Average(nave,noisy_objfn)
    xsol=solver.solve(x0,stoch_objfn)
    [xvals,fvals]=solver.get_stats()
    return [xsol,xvals,fvals]

test='Stochastic'
batches=[1,10,25,100]
#test='Momentum'
if test == 'Stochastic':
    iter_filename='Stochastic_Rosenbrock_iterations_noise{:}.png'.format(noise_mu)
    conv_filename='Stochastic_Rosenbrock_convergence_noise{:}.png'.format(noise_mu)
    #iter_filename=None
    #conv_filename=None
    options={'Gradient':{'use_hess':False},'Steplength estimator':{'use_hess':True,'use_estimate':True},'Newton':{'use_hess':True,'use_estimate':False}}
    colors=dict(zip(options.keys(),['orange','c','r']))
    lines=dict(zip(options.keys(),['k-','b-','r-']))
else:
    iter_filename='StochMom_Rosenbrock_iterations.png'
    conv_filename='StochMom_Rosenbrock_convergence.png'
    options={'Gradient':{'use_hess':False,'momentum':False},'Momentum':{'use_hess':False,'momentum':True}}
    #'Steplength est.(mom)':{'use_hess':True,'use_estimate':True,'momentum':True},
    #'Newton (mom)':{'use_hess':True,'use_estimate':False,'momentum':True}}
    colors=dict(zip(options.keys(),['orange','c','r','m']))
    lines=dict(zip(options.keys(),['k-','b-','r-','c-']))

#options={'stepEst':{'use_hess':True,'use_estimate':True}}
#options={'stepEst':{'use_hess':True,'use_estimate':True},'hess':{'use_hess':True,'use_estimate':False}}
#colors=dict(zip(options.keys(),['orange','c','r']))
#lines=dict(zip(options.keys(),['k-','b-','r-']))

for nb in batches:
    objs[nb]={}
    for option,args in options.items():
        [xsol,xvs,fvs]=solver(args,nb)
        xvals[option]=xvs
        objs[nb][option]=fvs
    print(objs[nb].keys())


plot=False
if plot:
    fig, ax = plt.subplots()
    cs = ax.contourf(xv, yv, zv,40)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    cbar = fig.colorbar(cs)

plot_convergence=False
if plot_convergence:
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
    if iter_filename:
        plt.savefig(iter_filename)   
    plt.show()

assert len(batches)==4
fig, ax=plt.subplots(2,2,squeeze=False, figsize=(18,10))
iters=np.arange(0,niter)

for nb, batch in enumerate(batches):
    i,j=divmod(nb,2)
    ax[i][j].set_title("Nbatches={:}".format(batch),fontsize=20)
    if i==1:
        ax[i][j].set_xlabel('Iterations')
    ax[i][j].set_ylabel('Error')
    ax[i][j].set_xlim([-0.1,niter+0.1])
    ax[i][j].set_ylim([0.001, 1.01])
    handles=[]
    for name,obj in objs[batch].items():
        print('line name:',name)
        ls=lines[name]
        if len(obj) < niter:
            obj0=[obj[-1]]*niter
            obj0[0:len(obj)]=np.array(obj)/obj[0]
        else:
            obj0=np.array(obj)/obj[0]
        ax[i][j].semilogy(iters, obj0, ls,label=name)
    #handles.append(line)
    ax[i][j].legend(loc='lower left')
if conv_filename:
    plt.savefig(conv_filename)
plt.show()
      
