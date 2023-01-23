# On the Role of Fixed Points of Dynamical Systems in Training Physics-Informed Neural Networks
Code to Paper: https://openreview.net/forum?id=56cTmVrg5w

## Abstract
This paper empirically studies commonly observed training difficulties of Physics-Informed Neural Networks (PINNs) on dynamical systems.
Our results indicate that fixed points which are inherent to these systems play a key role in the optimization of the in PINNs embedded physics loss function.
We observe that the loss landscape exhibits local optima that are shaped by the presence of fixed points.
We find that these local optima contribute to the complexity of the physics loss optimization which can explain common training difficulties and resulting nonphysical predictions.
Under certain settings, e.g., initial conditions close to fixed points or long simulations times, we show that those optima can even become better than that of the desired solution.


## Requirements
- tensorflow>2.2
- scipy
- pyDOE
- pyyaml
- pandas
- scikit-learn

## Information for Usage
The code is split into four section, resampling the four dynamical systems discussed in the paper: 

- Allen-Cahn
- Navier-Stokes
- Toy Example
- Undamped Pendulum

Each system folder contains a jupyter notebook (**main.ipynb**) that provides the main starting point for usage.
In that notebook, the configuration of the system, the Physics-Informed Neural Network (PINN) and its optimization is loaded in the beginning, which can be modified either by changing the ***.yaml** file in the *configs* folder or by passing a dictionary through *config_update* in the *load_config* function.
Based on the configuration settings, a PINN instance is initialized and trained. After training, a few plotting functions are provided to visualize the training behaviour and final PINN prediction, compared to the reference data. For the Navier-Stokes system, the reference data has to be downloaded first by running through the notebook (**download_DNS_data.ipynb**) in the *data* folder.

The settings in the **default.yaml** configuration - as well as many other configurations, as you might know from reading the paper - results in an **unsuccessful PINN training influenced by a fixed point inherent to the dynamical system**.

For more information, questions or requests, please drop an E-mail at [frohrhofer@acm.org](frohrhofer@acm.org).

## Citation
```
@article{rohrhofer2022on,
  title={On the Role of Fixed Points of Dynamical Systems in Training Physics-Informed Neural Networks},
  author={Franz M. Rohrhofer and Stefan Posch and Clemens G{\"o}{\ss}nitzer and Bernhard C Geiger},
  journal={Transactions on Machine Learning Research},
  year={2022},
  url={https://openreview.net/forum?id=56cTmVrg5w}
}
```

