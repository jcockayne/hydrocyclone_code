#include <functional>
#include "likelihood.hpp"
#include "pcn.hpp"
#include "simulate.hpp"
#include "memory_utils.hpp"
#include <iostream>

std::unique_ptr<SimulateResult> run_pcn_parallel(
	int n_iter,
	double beta,
	const Eigen::Ref<const Eigen::MatrixXd> &theta_0,
	const Eigen::Ref<const Eigen::MatrixXd> &sqrt_prior_cov,
	const Eigen::Ref<const Eigen::MatrixXd> &interior,
	const Eigen::Ref<const Eigen::MatrixXd> &sensors,
	const Eigen::Ref<const Eigen::MatrixXd> &theta_projection_mat,
	const Eigen::Ref<const Eigen::VectorXd> &kernel_args,
	const Eigen::Ref<const Eigen::MatrixXd> &stim_pattern,
	const Eigen::Ref<const Eigen::MatrixXd> &meas_pattern,
	const Eigen::Ref<const Eigen::MatrixXd> &data,
	double likelihood_variance,
	int n_threads
)
{
	auto log_likelihood_function = [
		interior, 
		sensors, 
		theta_projection_mat, 
		kernel_args, 
		stim_pattern, 
		meas_pattern, 
		data, 
		likelihood_variance
	] (const Eigen::VectorXd &theta) -> double {
		return log_likelihood(interior, sensors, theta, theta_projection_mat, kernel_args, stim_pattern, meas_pattern, data, likelihood_variance);
	};

	Eigen::MatrixXd ret_samples(theta_0.rows(), theta_0.cols());
	Eigen::VectorXd ret_likelihoods(theta_0.rows());
	#pragma omp parallel for num_threads(n_threads)
	for(int i = 0; i < theta_0.rows(); i++) {
		Eigen::VectorXd sample = theta_0.row(i);
		auto results = apply_one_kernel_pcn(sample, n_iter, beta, log_likelihood_function, sqrt_prior_cov, false);

		ret_samples.row(i) = results->result;
		ret_likelihoods(i) = results->log_likelihood;
	}

	return make_unique<SimulateResult>(ret_samples, ret_likelihoods);
}