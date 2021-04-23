#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gurobipy import *
import numpy as np
import math
from sklearn.gaussian_process.kernels import RBF
from scipy.optimize import minimize
#command to open console for testing 
#%qtconsole

#problem setup
def d_hat(p,x):
    return 10-p
def pi_1(x):
    return 3
def pi_0(x):
    return 6
X=3#the feature to evaluate
P=5.0#the price to evaluate
sigma=1#kernel bandwidth
c=10#bound on RKHS norm of f
P_grid=np.arange(1,10)#P_1,P_2,..,P_N
N=len(P_grid)
def f(p,x,w):
    beta0,beta1,alpha_vec=w[0],w[1],w[2:]
    f_val=beta1*math.exp(-((pi_1(x)-p)/sigma)**2)+beta0*math.exp(-((pi_0(x)-p)/sigma)**2)
    for i in range(0,N):
        f_val=f_val+alpha_vec[i]*math.exp(-((P_grid[i]-p)/sigma)**2)
    return f_val
def g(p,x):#return a vector of coefficients in f
    v=np.array([0.0]*(N+2))
    v[0]=math.exp(-((pi_0(x)-p)/sigma)**2)
    v[1]=math.exp(-((pi_1(x)-p)/sigma)**2)
    for i in range(2,N+2):
        v[i]=math.exp(-((P_grid[i-2]-p)/sigma)**2)
    return v

#pre-compute constants in the optimization problem
#constants in objective
v_pi_1=g(pi_1(X),X)
v_pi_0=g(pi_0(X),X)
#constants in constraints
v_arr=[[0.0 for i in range(N+2)] for j in range(N)]
for i in range(0,N):
    v_arr[i]=g(P_grid[i],X)
hat_d_grid=np.array([0.0]*N)
for i in range(0,N):
    hat_d_grid[i]=d_hat(P_grid[i],X)
#Gram matrix M
P_tilde=np.array([0.0]*(N+2))
P_tilde[0]=pi_0(X)
P_tilde[1]=pi_1(X)
P_tilde[2:]=P_grid
M=[[0.0 for i in range(N+2)] for j in range(N+2)]
for i in range(0,N+2):
    for j in range(0,N+2):
        M[i][j]=math.exp(-((P_tilde[i]-P_tilde[j])/sigma)**2)
        
#compute our bound
kappa=math.sqrt(2-2*math.exp(-((pi_0(X)-pi_1(X))/sigma)**2))
print('Our bound on (f(pi_1(X),X)-f(pi_0(X),X)) is: ',kappa*c)


# In[2]:


#(QP2) With monotonicity constraint
#build optimization model
m = Model()
m.Params.LogToConsole = 0#suppress Gurobipy printing
# Add variables
w=m.addVars ( N+2,name="w" )
# Set objective function
m.setObjective(quicksum(w[i] * (v_pi_1[i]-v_pi_0[i]) for i in range(N+2)), GRB.MAXIMIZE)
# Add constraints
m.addMQConstr (M, None, '<', c**2)#RKHS norm of f is no greater than c
m.addConstrs (hat_d_grid[i]-quicksum(v_arr[i][j]*w[j]/P_grid[i] for j in range(N+2))>=
              hat_d_grid[i+1]-quicksum(v_arr[i+1][j]*w[j]/P_grid[i+1] for j in range(N+2))
              for i in range(0,N-1))
#m.params.method=0  #0 is primal simplex, 1 is dual simplex, 2 is barrier
m.update()
m.optimize()

#print key info for testing purpose
obj = m.getObjective()
print('optimal objective value: ',obj.getValue())
w_star=np.array([0.0]*(N+2))
for i in range(0,N):
    w_star[i]=w[i].x
print('beta0,beta1,alpha: ',w_star[0],w_star[1],w_star[2:])
print('demand at Price=1,2,...,9: ')
for i in range(0,N):#print demand at grid points
    print(d_hat(P_grid[i],X)-f(P_grid[i],X,w_star)/P_grid[i])

#plot the function f
P_grid_2=np.arange(1,101)/10
Y=np.array([0.0]*len(P_grid_2))
hat_d_grid_2=np.array([0.0]*len(P_grid_2))
for i in range(0,len(P_grid_2)):
    hat_d_grid_2[i]=d_hat(P_grid_2[i],X) 
    Y[i]=f(P_grid_2[i],X,w_star)
from matplotlib import pyplot
pyplot.plot(P_grid_2, Y,label='worst case f')
pyplot.plot(P_grid_2, hat_d_grid_2,label='estimated demand (hat d)')
pyplot.plot(P_grid_2, hat_d_grid_2-np.divide(Y,P_grid_2),label='worst case demand')
pyplot.grid()
pyplot.xlabel('price')
pyplot.ylabel('')
pyplot.legend(loc='upper right')
pyplot.title('(QP2) worst case f and demand')
pyplot.show()


# In[3]:


#(QP3) without monotonicity constraint
#build optimization model
m = Model()
m.Params.LogToConsole = 0#suppress Gurobipy printing
# Add variables
w=m.addVars ( N+2,name="w" )
# Set objective function
m.setObjective(quicksum(w[i] * (v_pi_1[i]-v_pi_0[i]) for i in range(N+2)), GRB.MAXIMIZE)
# Add constraints
m.addMQConstr (M, None, '<', c**2)#RKHS norm of f is no greater than c
m.update()
m.optimize()

#print key info for testing purpose
obj = m.getObjective()
print('optimal objective value: ',obj.getValue())
w_star=np.array([0.0]*(N+2))
for i in range(0,N):
    w_star[i]=w[i].x
print('beta0,beta1,alpha: ',w_star[0],w_star[1],w_star[2:])
print('demand at Price=1,2,...,9: ')
for i in range(0,N):#print demand at grid points
    print(d_hat(P_grid[i],X)-f(P_grid[i],X,w_star)/P_grid[i])

#plot the function f
P_grid_2=np.arange(1,101)/10
Y=np.array([0.0]*len(P_grid_2))
hat_d_grid_2=np.array([0.0]*len(P_grid_2))
for i in range(0,len(P_grid_2)):
    hat_d_grid_2[i]=d_hat(P_grid_2[i],X) 
    Y[i]=f(P_grid_2[i],X,w_star)
from matplotlib import pyplot
pyplot.plot(P_grid_2, Y,label='worst case f')
pyplot.plot(P_grid_2, hat_d_grid_2,label='estimated demand (hat d)')
pyplot.plot(P_grid_2, hat_d_grid_2-np.divide(Y,P_grid_2),label='worst case demand')
pyplot.grid()
pyplot.xlabel('price')
pyplot.ylabel('')
pyplot.legend(loc='upper right')
pyplot.title('(QP3) worst case f and demand')
pyplot.show()


# In[ ]:




