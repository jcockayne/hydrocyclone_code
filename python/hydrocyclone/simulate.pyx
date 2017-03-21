from eigency.core cimport *
cimport numpy as np
import numpy as np
from cython.operator cimport dereference as deref
from libcpp.memory cimport unique_ptr

cdef extern from "simulate.hpp":
	cdef cppclass SimulateResult:
		MatrixXd samples
		VectorXd log_likelihoods
	cdef unique_ptr[SimulateResult] _run_pcn_parallel "run_pcn_parallel"(
		int n_iter,
		double beta,
		Map[MatrixXd] theta_0,
		Map[MatrixXd] sqrt_prior_cov,
		Map[MatrixXd] interior,
		Map[MatrixXd] sensors,
		Map[MatrixXd] theta_projection_mat,
		Map[VectorXd] kernel_args,
		Map[MatrixXd] stim_pattern,
		Map[MatrixXd] meas_pattern,
		Map[MatrixXd] data,
		double likelihood_variance,
		int n_threads
	)

def run_pcn_parallel(
	int n_iter,
	double beta,
	np.ndarray[ndim=2, dtype=np.float_t] theta_0,
	np.ndarray[ndim=2, dtype=np.float_t] sqrt_prior_cov,
	np.ndarray[ndim=2, dtype=np.float_t] interior,
	np.ndarray[ndim=2, dtype=np.float_t] sensors,
	np.ndarray[ndim=2, dtype=np.float_t] theta_projection_mat,
	np.ndarray[ndim=1, dtype=np.float_t] kernel_args,
	np.ndarray[ndim=2, dtype=np.float_t] stim_pattern,
	np.ndarray[ndim=2, dtype=np.float_t] meas_pattern,
	np.ndarray[ndim=2, dtype=np.float_t] data,
	double likelihood_variance,
	int n_threads
):
	ret = _run_pcn_parallel(
		n_iter,
		beta,
		Map[MatrixXd](theta_0),
		Map[MatrixXd](sqrt_prior_cov),
		Map[MatrixXd](interior),
		Map[MatrixXd](sensors),
		Map[MatrixXd](theta_projection_mat),
		Map[VectorXd](kernel_args),
		Map[MatrixXd](stim_pattern),
		Map[MatrixXd](meas_pattern),
		Map[MatrixXd](data),
		likelihood_variance,
		n_threads
	)

	return ndarray_copy(deref(ret).samples), ndarray_copy(deref(ret).log_likelihoods)