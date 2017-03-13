import bayesian_pdes as bpdes
import numpy as np
from scipy import stats
import mcmc


def theta_to_a(theta, sz_int, sz_bdy, proposal_dot_mat):
    theta = np.real_if_close(theta)
    theta_mod = np.dot(proposal_dot_mat, theta[:,None])
    kappa_int = theta_mod[:sz_int]
    kappa_bdy = theta_mod[sz_int:sz_int+sz_bdy]
    grad_kappa_x = theta_mod[sz_int+sz_bdy:2*sz_int+sz_bdy]
    grad_kappa_y = theta_mod[2*sz_int+sz_bdy:]
    return kappa_int, kappa_bdy, grad_kappa_x, grad_kappa_y


def construct_posterior(grid, op_system, theta, collocate_args, proposal_dot_mat, debug=False):
    design_int = grid.interior_plus_boundary
    a_int, a_bdy, a_x, a_y = theta_to_a(theta,
                                        design_int.shape[0],
                                        grid.sensors.shape[0],
                                        proposal_dot_mat
                                        )

    augmented_int = np.column_stack([design_int, a_int, a_x, a_y])
    augmented_bdy = np.column_stack([grid.sensors, a_bdy, np.nan * np.zeros((a_bdy.shape[0], 2))])
    obs = [
        (augmented_int, None),
        (augmented_bdy, None)
    ]
    posterior = bpdes.collocate(
        op_system.operators,
        op_system.operators_bar,
        obs,
        op_system,
        collocate_args,
        inverter='np'
    )
    return posterior


def phi(grid, op_system, theta, likelihood_variance, pattern, data, collocate_args, proposal_dot_mat, debug=False):
    # first solve forward
    design_int = grid.interior_plus_boundary
    posterior = construct_posterior(grid, op_system, theta, collocate_args, proposal_dot_mat, debug=debug)
    # now determine voltage at the sensor locations
    # we have seven observations so take one for each sensor other than sensor 1, the reference sensor
    augmented_locations = np.column_stack([grid.sensors, np.nan * np.zeros((8, 3))])
    mu_mult, Sigma = posterior.no_obs_posterior(augmented_locations)

    # now need to iterate the stim patterns and compute the residual
    rhs_int = np.zeros((len(design_int), 1))

    Sigma_obs = np.dot(pattern.meas_pattern, np.dot(Sigma, pattern.meas_pattern.T))
    likelihood_cov = Sigma_obs + likelihood_variance * np.eye(Sigma_obs.shape[0])
    # likelihood_cov = likelihood_variance*np.eye(Sigma_obs.shape[0])
    likelihood_dist = stats.multivariate_normal(np.zeros(Sigma_obs.shape[0]), likelihood_cov)

    if debug:
        print("Sigma diag: {}\tCondition:{} \t Augmented Condition: {}".format(np.diag(Sigma), np.linalg.cond(Sigma),
                                                                               np.linalg.cond(likelihood_cov)))

    likelihood = 0
    for voltage, current in zip(data, pattern.stim_pattern):
        rhs_bdy = current[:, None]
        rhs = np.row_stack([rhs_int, rhs_bdy])

        model_voltage = np.dot(pattern.meas_pattern, np.dot(mu_mult, rhs))

        residual = voltage.ravel() - model_voltage.ravel()
        this_likelihood = likelihood_dist.logpdf(residual)
        if debug:
            print("Model|True\n {}".format(np.c_[model_voltage, voltage]))
            print("Likelihood: {}   |   Residual: {}".format(this_likelihood, np.abs(residual).sum()))
        likelihood += this_likelihood
    return -likelihood

class PCNKernel(object):
    def __init__(self, proposal, grid, op_system, likelihood_variance, pattern, data, collocate_args, proposal_dot_mat):
        self.__proposal__ = proposal   
        self.__grid__ = grid
        self.__op_system__ = op_system
        self.__likelihood_variance__ = likelihood_variance
        self.__pattern__ = pattern
        self.__data__ = data
        self.__collocate_args__ = collocate_args
        self.__proposal_dot_mat__ = proposal_dot_mat

    def phi(self, theta, debug=False):
        return phi(
            self.__grid__,
            self.__op_system__,
            theta,
            self.__likelihood_variance__,
            self.__pattern__,
            self.__data__,
            self.__collocate_args__,
            self.__proposal_dot_mat__,
            debug
        )

    def apply_pcn(self, kappa_0, n_iter):
        return mcmc.pCN(n_iter, self.__proposal__, self.phi, kappa_0)