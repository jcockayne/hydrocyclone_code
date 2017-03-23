#include <Eigen/Core>
#ifndef LIKELIHOOD_H

double log_likelihood(
	const Eigen::Ref<const Eigen::MatrixXd> &interior,
	const Eigen::Ref<const Eigen::MatrixXd> &sensors,
	const Eigen::Ref<const Eigen::VectorXd> &theta,
	const Eigen::Ref<const Eigen::MatrixXd> &theta_projection_mat,
	const Eigen::Ref<const Eigen::VectorXd> &kernel_args,
	const Eigen::Ref<const Eigen::MatrixXd> &stim_pattern,
	const Eigen::Ref<const Eigen::MatrixXd> &meas_pattern,
	const Eigen::Ref<const Eigen::MatrixXd> &data,
	double likelihood_variance,
	bool debug = false
);

#define LIKELIHOOD_H
#endif