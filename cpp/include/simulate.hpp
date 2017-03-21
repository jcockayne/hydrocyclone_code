#ifndef SIMULATE_H

struct SimulateResult {
	const Eigen::MatrixXd samples;
	const Eigen::VectorXd acceptances;
	const Eigen::VectorXd log_likelihoods;

	SimulateResult(const Eigen::MatrixXd samples, const Eigen::VectorXd acceptances, const Eigen::VectorXd log_likelihoods) 
		: samples(samples), acceptances(acceptances), log_likelihoods(log_likelihoods) { }
};

std::unique_ptr<SimulateResult> run_pcn_parallel(
	int n_iter,
	double beta,
	const Eigen::Ref<const Eigen::MatrixXd> &theta_0,
	const Eigen::Ref<const Eigen::VectorXd> &prior_mean,
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
);

#endif