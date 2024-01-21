# FCDRL_TSCOPF
This repository contains the whole implementation of the **Fast-Converged Deep Reinforcement Learning for Optimal Dispatch of Large-Scale Power Systems under Transient Security Constraints.** 
The code is developed using [Py_PSOPS](https://github.com/xxh0523/Py_PSOPS). 

# Preparation
Just download the whole repository. The code can run on Windows and Linux platforms.

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

# References
[1] **T. Xiao**, Y. Chen*, H. Diao, S. Huang, C. Shen, “On Fast-Converged Deep Reinforcement Learning for Optimal Dispatch of Large-Scale Power Systems under Transient Security Constraints,” [arxiv](https://arxiv.org/abs/2304.08320)

[2] **T. Xiao**, Y. Chen*, J. Wang, S. Huang, W. Tong, and T. He, “Exploration of AI-Oriented Power System Transient Stability Simulations,” *Journal of Modern Power Systems and Clean Energy*, vol. 11, no. 2, pp. 401–411, Mar. 2023, doi: [10.35833/MPCE.2022.000099](https://ieeexplore.ieee.org/document/9833418)
