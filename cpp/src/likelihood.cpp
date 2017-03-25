#include <Eigen/Dense>
#include <math.h>
#include "collocate.hpp"
#include "likelihood.hpp"
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
	return log_likelihood(
		interior,
		sensors,
		theta,
		theta_projection_mat,
		kernel_args,
		stim_pattern,
		meas_pattern,
		data,
		likelihood_variance,
		NULL,
		debug
	);
}

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
	Collocator *collocator,
	bool debug
)
{
	return log_likelihood_tempered(
		interior,
		sensors,
		theta,
		theta_projection_mat,
		kernel_args,
		stim_pattern,
		meas_pattern,
		data,
		Eigen::MatrixXd(0, 0),
		0.,
		likelihood_variance,
		collocator,
		debug
	);
}

double log_likelihood_tempered(
	const Eigen::Ref<const Eigen::MatrixXd> &interior,
	const Eigen::Ref<const Eigen::MatrixXd> &sensors,
	const Eigen::Ref<const Eigen::VectorXd> &theta,
	const Eigen::Ref<const Eigen::MatrixXd> &theta_projection_mat,
	const Eigen::Ref<const Eigen::VectorXd> &kernel_args,
	const Eigen::Ref<const Eigen::MatrixXd> &stim_pattern,
	const Eigen::Ref<const Eigen::MatrixXd> &meas_pattern,
	const Eigen::Ref<const Eigen::MatrixXd> &data_1,
	const Eigen::Ref<const Eigen::MatrixXd> &data_2,
	double temperature,
	double likelihood_variance,
	bool debug
)
{
	return log_likelihood_tempered(
		interior,
		sensors,
		theta,
		theta_projection_mat,
		kernel_args,
		stim_pattern,
		meas_pattern,
		data_1,
		data_2,
		temperature,
		likelihood_variance,
		NULL,
		debug
	);
}

double log_likelihood_tempered(
	const Eigen::Ref<const Eigen::MatrixXd> &interior,
	const Eigen::Ref<const Eigen::MatrixXd> &sensors,
	const Eigen::Ref<const Eigen::VectorXd> &theta,
	const Eigen::Ref<const Eigen::MatrixXd> &theta_projection_mat,
	const Eigen::Ref<const Eigen::VectorXd> &kernel_args,
	const Eigen::Ref<const Eigen::MatrixXd> &stim_pattern,
	const Eigen::Ref<const Eigen::MatrixXd> &meas_pattern,
	const Eigen::Ref<const Eigen::MatrixXd> &data_1,
	const Eigen::Ref<const Eigen::MatrixXd> &data_2,
	double temperature,
	double likelihood_variance,
	Collocator *collocator,
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
	
	std::unique_ptr<CollocationResult> posterior;
	if(collocator != NULL)
		posterior = collocator->collocate_no_obs(augmented_sens, augmented_int, augmented_sens, kernel_args);
	else
		posterior = collocate_no_obs(augmented_sens, augmented_int, augmented_sens, kernel_args);

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
	double halflogdet = 0;
	for(int i = 0; i < L.rows(); i++)
		halflogdet += log(L(i,i));
	double halflog2pi = 0.5*log(2*M_PI);
	// NB ASSUMES THAT data_1 AND data_2 HAVE THE SAME ROWS!
	double log_norm_const = -halflog2pi*data_1.cols() - halflogdet;

	#ifdef WITH_DEBUG
	if(debug)
		std::cout << log_norm_const << std::endl;
	#endif

	Eigen::MatrixXd left_model_mult = meas_pattern * posterior->mu_mult;
	double likelihood_1 = 0;
	double likelihood_2 = 0;
	if(temperature < 1 && data_1.rows() > 0) {
		for(int i = 0; i < data_1.rows(); i++) {
			rhs.bottomRows(stim_pattern.cols()) = stim_pattern.row(i).transpose();
			Eigen::VectorXd residual = left_model_mult*rhs - data_1.row(i).transpose();

			double this_likelihood = -0.5*residual.dot(likelihood_cov_decomp.solve(residual)) + log_norm_const;
			likelihood_1 += this_likelihood;
			#ifdef WITH_DEBUG
			if(debug)
				std::cout << this_likelihood << std::endl;
			#endif
		}
	}


	log_norm_const = -halflog2pi*data_2.cols() - halflogdet;
	if(temperature > 0 && data_2.rows() > 0) {
		for(int i = 0; i < data_2.rows(); i++) {
			rhs.bottomRows(stim_pattern.cols()) = stim_pattern.row(i).transpose();
			Eigen::VectorXd residual = left_model_mult*rhs - data_2.row(i).transpose();

			double this_likelihood = -0.5*residual.dot(likelihood_cov_decomp.solve(residual)) + log_norm_const;
			likelihood_2 += this_likelihood;
			#ifdef WITH_DEBUG
			if(debug)
				std::cout << this_likelihood << std::endl;
			#endif
		}
	}
	return likelihood_1*(1-temperature) + likelihood_2*temperature;
}