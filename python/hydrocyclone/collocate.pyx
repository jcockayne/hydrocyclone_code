from eigency.core cimport *
from libcpp.memory cimport unique_ptr
cimport numpy as np
from cython.operator cimport dereference as deref

cdef extern from "collocate.hpp":
	cdef cppclass CollocationResult:
		MatrixXd mu_mult
		MatrixXd cov
		CollocationResult(MatrixXd mu_mult, MatrixXd cov)
	cdef unique_ptr[CollocationResult] _collocate_no_obs "collocate_no_obs"(
		Map[MatrixXd] x,
		Map[MatrixXd] interior,
		Map[MatrixXd] sensors,
		Map[VectorXd] kernel_args
	)

def collocate_no_obs(
	np.ndarray[dtype=np.float_t, ndim=2] x,
	np.ndarray[dtype=np.float_t, ndim=2] interior,
	np.ndarray[dtype=np.float_t, ndim=2] sensors,
	np.ndarray[dtype=np.float_t, ndim=1] kernel_args
):
	cdef unique_ptr[CollocationResult] ret = _collocate_no_obs(
		Map[MatrixXd](x),
		Map[MatrixXd](interior),
		Map[MatrixXd](sensors),
		Map[VectorXd](kernel_args)
	)
	mu_mult = ndarray_copy(deref(ret).mu_mult)
	cov = ndarray_copy(deref(ret).cov)
	return mu_mult, cov