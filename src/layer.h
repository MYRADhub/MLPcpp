#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace Layers {
	class Layer {
	public:
		virtual Eigen::VectorXd forward(const Eigen::VectorXd& input) = 0;
		virtual Eigen::VectorXd backward(const Eigen::VectorXd& grad_output, double learning_rate) = 0;
	};

	class Activation : public Layer {
	private:
		Eigen::VectorXd input_cache;
		std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation_fn;
		std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation_prime_fn;

	public:
		Activation(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> fn,
			std::function<Eigen::VectorXd(const Eigen::VectorXd&)> prime_fn)
			: activation_fn(fn), activation_prime_fn(prime_fn) {}

		Eigen::VectorXd forward(const Eigen::VectorXd& input) override {
			input_cache = input;
			return activation_fn(input);
		}

		Eigen::VectorXd backward(const Eigen::VectorXd& grad_output, double learning_rate = 0.0) override {
			Eigen::VectorXd grad_input = grad_output.array() * activation_prime_fn(input_cache).array();
			return grad_input;
		}
	};

	class Linear : public Layer {
	private:
		Eigen::MatrixXd weights;
		Eigen::VectorXd biases;
		Eigen::MatrixXd input_cache; // Cache for backward pass

	public:
		Linear(int input_size, int output_size)
			: weights(Eigen::MatrixXd::Random(output_size, input_size)* std::sqrt(1.0 / input_size)),
			biases(Eigen::VectorXd::Random(output_size)) {}

		Eigen::VectorXd forward(const Eigen::VectorXd& input) override {
			input_cache = input;
			return (weights * input) + biases;
		}

		Eigen::VectorXd backward(const Eigen::VectorXd& grad_output, double learning_rate) override {
			Eigen::VectorXd grad_input = weights.transpose() * grad_output;
			Eigen::MatrixXd grad_weights = grad_output * input_cache.transpose();
			Eigen::VectorXd grad_biases = grad_output;

			// Gradient clipping
			//grad_weights = grad_weights.cwiseMin(1.0).cwiseMax(-1.0);
			//grad_biases = grad_biases.cwiseMin(1.0).cwiseMax(-1.0);

			// Update weights and biases (SGD for simplicity, might add other optimizer support in future)
			weights -= learning_rate * grad_weights;
			biases -= learning_rate * grad_biases;

			return grad_input;
		}
	};

	class Conv1d : public Layer {
	private:
		int in_channels, out_channels;
		int kernel_size, stride, padding;
		Eigen::MatrixXd kernels;
		Eigen::VectorXd biases;
		Eigen::MatrixXd input_cache;

	public:
		Conv1d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0)
			: in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), stride(stride), padding(padding),
			kernels(Eigen::MatrixXd::Random(out_channels, in_channels * kernel_size)),
			biases(Eigen::VectorXd::Random(out_channels)) {}

		Eigen::VectorXd forward(const Eigen::VectorXd& input) override {
			input_cache = Eigen::MatrixXd::Zero(input.size() + 2 * padding, 1);
			input_cache.block(padding, 0, input.size(), 1) = input;

			int output_size = (input_cache.rows() - kernel_size) / stride + 1;
			Eigen::VectorXd output = Eigen::VectorXd::Zero(out_channels * output_size);

			for (int oc = 0; oc < out_channels; oc++) {
				for (int i = 0; i < output_size; i++) {
					Eigen::VectorXd region = input_cache.block(i * stride, 0, kernel_size, 1);
					double value = (kernels.row(oc).head(region.size()).dot(region)) + biases[oc];
					output(oc * output_size + i) = value;
				}
			}
			return output;
		}

		Eigen::VectorXd backward(const Eigen::VectorXd& grad_output, double learning_rate) override {
			int output_size = grad_output.size() / out_channels;
			Eigen::MatrixXd grad_kernels = Eigen::MatrixXd::Zero(out_channels, in_channels * kernel_size);
			Eigen::VectorXd grad_biases = Eigen::VectorXd::Zero(out_channels);
			Eigen::VectorXd grad_input = Eigen::VectorXd::Zero(input_cache.size() - 2 * padding);

			for (int oc = 0; oc < out_channels; ++oc) {
				for (int i = 0; i < output_size; ++i) {
					Eigen::VectorXd region = input_cache.block(i * stride, 0, kernel_size, 1);
					double grad_value = grad_output(oc * output_size + i);

					// Gradients for kernel and input
					grad_kernels.row(oc) += grad_value * region.transpose();
					grad_input.block(i * stride, 0, kernel_size, 1) += grad_value * kernels.row(oc).transpose();

					// Gradient for bias
					grad_biases[oc] += grad_value;
				}
			}

			// Update kernels and biases
			kernels -= learning_rate * grad_kernels;
			biases -= learning_rate * grad_biases;

			return grad_input.segment(padding, grad_input.size() - 2 * padding);
		}
	};

}