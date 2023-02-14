# Off-Policy Pricing Evaluation

This repository contains code pertaining to the paper "Off-Policy Pricing Evaluation" by A. Elmachtoub, V. Gupta and Y. Zhao.  

We consider a personalized pricing problem in which we have data consisting of feature information, historical pricing decisions, and binary realized demand. The goal is to perform off-policy evaluation for a new personalized pricing policy that maps features to prices. Methods based on inverse propensity weighting (including doubly robust methods) for off-policy evaluation may perform poorly when the logging policy has little exploration or is deterministic, which is common in pricing applications. Building on the balanced policy evaluation framework of Kallus (2018), we propose a new approach tailored to pricing applications. The key idea is to compute an estimate that either  minimizes the worst-case mean squared error or ii) maximizes a worst-case bound on policy performance, where in both cases the worst-case is taken with respect to a set of possible revenue functions. We establish theoretical convergence guarantees and empirically demonstrate the advantage of our approach using a real-world pricing dataset.

## Installation
Install the required packages listed below and the latest version of Gurobi optimization solver.
```
pip install sklearn
pip install scipy
pip install matplotlib 
pip install pandas
```

## Running Experiments

The following commands replicate the experiments on the Nomis dataset. 

```
python main.py --Nomis=True --new_policy=0.9
python main.py --Nomis=True --new_policy=0.95
python main.py --Nomis=True --new_policy=1.05
python main.py --Nomis=True --new_policy=1.1
```

The following commands replicate the experiments on synthetic datasets. 
```
python main.py --synthetic_new_policy=2
python main.py --synthetic_new_policy=3
python main.py --synthetic_new_policy=4
```
