# Off-Policy Pricing Evaluation

This repository contains code pertaining to the paper "Off-Policy Pricing Evaluation" by A. Elmachtoub, V. Gupta and Y. Zhao.  

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
