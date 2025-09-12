# FCDRL_TSCOPF
This repository contains the complete implementation of the **Fast-Converged Deep Reinforcement Learning for Optimal Dispatch of Large-Scale Power Systems under Transient Security Constraints.** 
The code is developed using [Py_PSOPS](https://github.com/xxh0523/Py_PSOPS). 

# Preparation
Simply download the whole repository. The code can run on Windows and Linux platforms.

# Requirements
Install the following packages before running the code. 
```
conda install qtwebkit
conda install numpy
pip install torch
pip install scikit-opt
pip install ray
pip install timebudget
```

# Agent training
With the default settings of hyperparameters, an agent can be trained with the following command.
```
python sopf_base.py --training
```

After training, the obtained agent will be saved at:
```
./00saved_results/models/scopf_agent/test/final_model_pth
```

For more details, please refer to the argparse in ```sopf_base.py```.

# Agent testing
Use the following command to test the agent:
```
python sopf_base.py --comparing --model .\00saved_results\models\scopf_agent\test\final_model.pth --round 10000
```

# Agent vs. PSO
Use the following command to compare the agent with the Particle Swarm Optimization (PSO) algorithm:
```
python sopf_base.py --comparing --model .\00saved_results\models\scopf_agent\test\final_model.pth --round 1
```
**Caution: PSO will take a very long time!**

# Parallelization
By adding ```--worker x```, multiple processors can be used for training, tesing, and comparing. 

For training, the command ```--worker x``` enables parallel exploration. For example, the following command will sample 4 scenarios simultaneously for exploration during training:
```
python sopf_base.py --training --worker 4
``` 

For testing and comparing, the command ```--worker x``` enables parallel decision-making for multiple scenarios at the same time. For example, the following command will test 4 scenarios at the same time, with each processor testing/comparing 2500 scenarios:
```
python sopf_base.py --comparing --model .\00saved_results\models\scopf_agent\test\final_model.pth --round 10000 --worker 4
```

# References
[1] **T. Xiao**, Y. Chen, H. Diao, S. Huang, and C. Shen, “Fast-Converging Deep Reinforcement Learning for Optimal Dispatch of Large-Scale Power Systems under Transient Security Constraints,” *Journal of Modern Power Systems and Clean Energy*, pp. 1–12, 2025, Early Access, doi: [10.35833/MPCE.2024.000624](https://ieeexplore.ieee.org/document/10944544).

[2] **T. Xiao**, Y. Chen*, J. Wang, S. Huang, W. Tong, and T. He, “Exploration of AI-Oriented Power System Transient Stability Simulations,” *Journal of Modern Power Systems and Clean Energy*, vol. 11, no. 2, pp. 401–411, Mar. 2023, doi: [10.35833/MPCE.2022.000099](https://ieeexplore.ieee.org/document/9833418).
