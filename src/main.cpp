#include "activation.h"
#include "layer.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <memory>

class MLP {
private:
	std::vector<std::unique_ptr<Layers::Layer>> layers;

public:
	void add_layer(int input_size, int output_size) {
		layers.push_back(std::make_unique<Layers::Linear>(input_size, output_size));
	}

	void add_activation(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation_fn,
		std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation_prime_fn) {
		layers.push_back(std::make_unique<Layers::Activation>(activation_fn, activation_prime_fn));
	}

	Eigen::VectorXd forward(const Eigen::VectorXd& input) {
		Eigen::VectorXd output = input;
		for (const auto& layer : layers) {
			output = layer->forward(output);
		}
		return output;
	}

	void backward(const Eigen::VectorXd& grad_output, double learning_rate = 0.01) {
		Eigen::VectorXd grad = grad_output;
		for (auto it = layers.rbegin(); it != layers.rend(); it++) {
			grad = (*it)->backward(grad, learning_rate);
		}
	}
};

int main(int argc, char **argv) {
	MLP model;

	model.add_layer(3, 6);
	model.add_activation(Activations::relu, Activations::relu_prime);
	model.add_layer(6, 1);
	model.add_activation(Activations::sigmoid, Activations::sigmoid_prime);

	// Training data (a U b -> c problem)
	std::vector<Eigen::VectorXd> inputs = {
		(Eigen::VectorXd(3) << 0, 0, 0).finished(),
		(Eigen::VectorXd(3) << 0, 0, 1).finished(),
		(Eigen::VectorXd(3) << 0, 1, 0).finished(),
		(Eigen::VectorXd(3) << 0, 1, 1).finished(),
		(Eigen::VectorXd(3) << 1, 0, 0).finished(),
		(Eigen::VectorXd(3) << 1, 0, 1).finished(),
		(Eigen::VectorXd(3) << 1, 1, 0).finished(),
		(Eigen::VectorXd(3) << 1, 1, 1).finished()
	};

	std::vector<Eigen::VectorXd> targets = {
		(Eigen::VectorXd(1) << 1).finished(),
		(Eigen::VectorXd(1) << 1).finished(),
		(Eigen::VectorXd(1) << 0).finished(),
		(Eigen::VectorXd(1) << 1).finished(),
		(Eigen::VectorXd(1) << 0).finished(),
		(Eigen::VectorXd(1) << 1).finished(),
		(Eigen::VectorXd(1) << 0).finished(),
		(Eigen::VectorXd(1) << 1).finished()
	};

	// Training hyperparameters
	int epochs = 1000;
	double learning_rate = 1e-2;
	std::cout << "Starting training\n";
	// Training loop
	for (int epoch = 0; epoch < epochs; epoch++) {
		double total_loss = 0;

		for (size_t i = 0; i < inputs.size(); i++) {
			Eigen::VectorXd prediction = model.forward(inputs[i]);

			Eigen::VectorXd error = prediction - targets[i];
			total_loss += error.squaredNorm();

			model.backward(error, learning_rate);
		}

		if (epoch % 10 == 0) {
			std::cout << "Epoch " << epoch << ", Loss: " << total_loss / inputs.size() << '\n';
		}
	}

	// Test the trained model
	std::cout << "\nTesting the MLP\n";
	for (size_t i = 0; i < inputs.size(); i++) {
		Eigen::VectorXd prediction = model.forward(inputs[i]);
		std::cout << "Input: " << inputs[i].transpose()
			<< ", Predicted: " << prediction.transpose()
			<< ", Target: " << targets[i].transpose() << std::endl;
	}

	std::cin.get();
}