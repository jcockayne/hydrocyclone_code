#include<Eigen/Core>
#ifndef OPERATORS_H

Eigen::MatrixXd Id_Id(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args);
Eigen::MatrixXd Id_A(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args);
Eigen::MatrixXd Id_B(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args);
Eigen::MatrixXd A_A(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args);
Eigen::MatrixXd A_B(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args);
Eigen::MatrixXd B_B(const Eigen::Ref<const Eigen::MatrixXd> &xarg, const Eigen::Ref<const Eigen::MatrixXd> &yarg, const Eigen::Ref<const Eigen::VectorXd> &args);

#define OPERATORS_H
#endif