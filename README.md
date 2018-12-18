# Bayesian Probabilistic Numerical Methods in Time-Dependent State Estimation for Industrial Hydrocyclone Equipment

This repository contains the code for the paper **Bayesian Probabilistic Numerical Methods in Time-Dependent State Estimation for Industrial Hydrocyclone Equipment** (Chris J. Oates, Jon Cockayne, Robert G. Ackroyd, Mark Girolami) [\[`arXiv`\]](https://arxiv.org/abs/1707.06107).

## Dependencies

The code has been run on a python 3 install and has not been tested on python 2.
Reproducing the experiments will require that the following standard python libraries are installed:

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `sympy`
- `jupyter`
- `pandas`

In addition, the python libraries contained in the following git repositories must be installed:

- [`bayesian_pdes`](https://github.com/jcockayne/bayesian_pdes)
- [`bayesian_eit`](https://github.com/jcockayne/bayesian_eit)

## Code Structure

With the above dependencies installed, nothing further needs to be installed to run the code.
All of the simulations are performed by Jupyter notebooks contained in the `notebooks` subdirectory.
There are two main categories of notebook: those that perform simulations and those that process results.
The notebooks that perform simulations are:

- **Hydrocyclone_Static_EIT**: Runs a static recovery simulation for a fixed time point, using the preconditioned Crank-Nicholson algorithm to perform the MCMC. The temporal component is not considered.
- **Hydrocyclone_Temporal_Recovery**: Runs the full temporal recovery that was used to generate the bulk of the results in the paper.

The remaining notebooks are for results processing. 

### Generic Results

- **Hydrocyclone_Results_Designs** plots the experimental designs used to solve both forward and inverse problems.

### Static Recovery Results
- **Hydrocyclone_Results_Static_Means** plots the posterior mean and standard deviation.
- **Hydrocyclone_Results_Static_PCs** plots the posterior in principle component directions.
- **Hydrocyclone_Results_Static_Variance** plots the change in integrated standard deviation as a function of the forward solver resolution.


### Temporal Recovery Results
- **Hydrocyclone_Results_Temporal** plots the evolution of the posterior mean and integrated standard deviation over time.
- **Hydrocyclone_Results_Temporal_Lambdas** examines the influence of the temporal smoothness parameter $\lambda$.
- **Hydrocyclone_Results_Temporal_PCs** plots the posterior in principle component directions as of the final time point.
- **Hydrocyclone_Results_Temporal_Variance** shows the change in the integrated standard deviation at the final time point as a function of the number of design points.

## Acknowledgements

The collection of the real tomographic data was supported by an EPSRC research grant (GR/R22148/01).
