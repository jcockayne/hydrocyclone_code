#include "operators.hpp"
#include "collocate.hpp"
#include <Eigen/Dense>
#include "memory_utils.hpp"
#include <iostream>
#include <fstream>


std::unique_ptr<CollocationResult> collocate_no_obs(
	const Eigen::Ref<const Eigen::MatrixXd> &x,
	const Eigen::Ref<const Eigen::MatrixXd> &interior, 
	const Eigen::Ref<const Eigen::MatrixXd> &sensors,
	const Eigen::Ref<const Eigen::VectorXd> &kernel_args
)
{
	int N = interior.rows() + sensors.rows();
	int M = x.rows();

	// TODO: allocating a bunch of temp storage here which could be re-used
	Eigen::MatrixXd left(M, N);
	Eigen::MatrixXd central(N, N);
	Eigen::MatrixXd kern(M, M);

	Id_A(x, interior, kernel_args, left.leftCols(interior.rows()));
	Id_B(x, sensors, kernel_args, left.rightCols(sensors.rows()));

	A_A(interior, interior, kernel_args, central.topLeftCorner(interior.rows(), interior.rows()));
	A_B(interior, sensors, kernel_args, central.topRightCorner(interior.rows(), sensors.rows()));
	B_B(sensors, sensors, kernel_args, central.bottomRightCorner(sensors.rows(), sensors.rows()));
	central.bottomLeftCorner(sensors.rows(), interior.rows()) = central.topRightCorner(interior.rows(), sensors.rows()).transpose();

	Id_Id(x, x, kernel_args, kern);

	/*
	std::ofstream file1("central.txt");
	file1 << central;
	std::ofstream file2("left.txt");
	file2 << left;
	std::ofstream file3("kern.txt");
	file3 << kern;
	*/
	// and lastly build the posterior
	// first invert the central matrix...
	Eigen::MatrixXd tmp = central.ldlt().solve(left.transpose());
	Eigen::MatrixXd mu_mult = tmp.transpose();
	Eigen::MatrixXd cov = kern - mu_mult * left.transpose();

	return make_unique<CollocationResult>(mu_mult, cov);
}