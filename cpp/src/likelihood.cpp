#include <Eigen/Dense>
#include <math.h>
#include "collocate.hpp"
#include <iostream>
#include <fstream>

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
	bool debug
)
{
	Eigen::VectorXd projected_theta = theta_projection_mat*theta;
	Eigen::VectorXd theta_int = projected_theta.topRows(interior.rows());
	Eigen::VectorXd theta_sens = projected_theta.segment(interior.rows(), sensors.rows());
	Eigen::VectorXd theta_x = projected_theta.segment(interior.rows() + sensors.rows(), interior.rows());
	Eigen::VectorXd theta_y = projected_theta.bottomRows(interior.rows());

	// augment the interior, sensors with theta
	Eigen::MatrixXd augmented_int(interior.rows(), 5);
	augmented_int << interior, theta_int, theta_x, theta_y;
	Eigen::MatrixXd augmented_sens(sensors.rows(), 5);
	augmented_sens << sensors, theta_sens;
	

	auto posterior = collocate_no_obs(augmented_sens, augmented_int, augmented_sens, kernel_args);

	Eigen::VectorXd rhs = Eigen::VectorXd::Zero(posterior->mu_mult.cols());

	Eigen::MatrixXd likelihood_cov = meas_pattern*posterior->cov*meas_pattern.transpose();

	for(int i = 0; i < likelihood_cov.rows(); i++)
		likelihood_cov(i,i) += likelihood_variance;

	/*
	std::ofstream file1("likelihood_cov.txt");
	file1 << likelihood_cov;
	*/

	auto likelihood_cov_decomp = likelihood_cov.llt();
	Eigen::MatrixXd L = likelihood_cov_decomp.matrixL();
	double logdet = 0;
	for(int i = 0; i < L.rows(); i++)
		logdet += log(L(i,i));
	logdet*=2;
	double log_norm_const = -0.5*data.cols()*log(2*M_PI) - 0.5*logdet;

	#ifdef WITH_DEBUG
	if(debug)
		std::cout << log_norm_const << std::endl;
	#endif

	double likelihood = 0;
	Eigen::MatrixXd left_model_mult = meas_pattern * posterior->mu_mult;
	for(int i = 0; i < data.rows(); i++) {
		rhs.bottomRows(stim_pattern.cols()) = stim_pattern.row(i).transpose();
		Eigen::VectorXd residual = left_model_mult*rhs - data.row(i).transpose();

		double this_likelihood = -0.5*residual.dot(likelihood_cov_decomp.solve(residual)) + log_norm_const;
		likelihood += this_likelihood;
		#ifdef WITH_DEBUG
		if(debug)
			std::cout << this_likelihood << std::endl;
		#endif
		/*
		{
			std::cout << "MODEL | TRUE | RESIDUAL: " << std::endl;
			Eigen::MatrixXd tmp(residual.rows(), 3);
			tmp << left_model_mult*rhs, data.row(i).transpose(), residual;
			std::cout << tmp << std::endl;
			std::cout << "LIKELIHOOD: " << this_likelihood << std::endl;
		}
		*/
	}

	return likelihood;
}