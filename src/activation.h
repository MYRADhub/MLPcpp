#pragma once
#include <Eigen/Dense>

namespace Activations {

	inline Eigen::VectorXd  relu(const Eigen::VectorXd& x) {
		return x.cwiseMax(0.0);
	}

	inline Eigen::VectorXd relu_prime(const Eigen::VectorXd& x) {
		return (x.array() > 0.0).cast<double>();
	}

	inline Eigen::VectorXd sigmoid(const Eigen::VectorXd& x) {
		return 1.0 / (1.0 + (-x.array()).exp());
	}

	inline Eigen::VectorXd sigmoid_prime(const Eigen::VectorXd& x) {
		Eigen::VectorXd s = sigmoid(x);
		return s.array() * (1 - s.array());
	}

	inline Eigen::VectorXd tanh_fn(const Eigen::VectorXd& x) {
		return x.array().tanh();
	}

	inline Eigen::VectorXd tanh_prime(const Eigen::VectorXd& x) {
		return 1.0 - x.array().tanh().square();
	}

}