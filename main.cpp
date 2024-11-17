#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <memory>

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

Eigen::VectorXd  relu(const Eigen::VectorXd& x) {
	return x.cwiseMax(0.0);
}

Eigen::VectorXd relu_prime(const Eigen::VectorXd& x) {
	return (x.array() > 0.0).cast<double>();
}

Eigen::VectorXd sigmoid(const Eigen::VectorXd& x) {
	return 1.0 / (1.0 + (-x.array()).exp());
}

Eigen::VectorXd sigmoid_prime(const Eigen::VectorXd& x) {
	Eigen::VectorXd s = sigmoid(x);
	return s.array() * (1 - s.array());
}

Eigen::VectorXd tanh_fn(const Eigen::VectorXd& x) {
	return x.array().tanh();
}

Eigen::VectorXd tanh_prime(const Eigen::VectorXd& x) {
	return 1.0 - x.array().tanh().square();
}

class Linear : public Layer {
private:
	Eigen::MatrixXd weights;
	Eigen::VectorXd biases;
	Eigen::MatrixXd input_cache; // Cache for backward pass

public:
	Linear(int input_size, int output_size)
		: weights(Eigen::MatrixXd::Random(output_size, input_size) * std::sqrt(1.0 / input_size)),
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

class MLP {
private:
	std::vector<std::shared_ptr<Layer>> layers;

public:
	void add_layer(int input_size, int output_size) {
		layers.push_back(std::make_shared<Linear>(input_size, output_size));
	}

	void add_activation(std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation_fn,
		std::function<Eigen::VectorXd(const Eigen::VectorXd&)> activation_prime_fn) {
		layers.push_back(std::make_shared<Activation>(activation_fn, activation_prime_fn));
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

int main() {
	MLP model;

	model.add_layer(2, 6);
	model.add_activation(relu, relu_prime);
	model.add_layer(6, 1);
	model.add_activation(sigmoid, sigmoid_prime);

	// Training data (XOR problem)
	std::vector<Eigen::VectorXd> inputs = {
		(Eigen::VectorXd(2) << 0, 0).finished(),
		(Eigen::VectorXd(2) << 0, 1).finished(),
		(Eigen::VectorXd(2) << 1, 0).finished(),
		(Eigen::VectorXd(2) << 1, 1).finished(),
	};

	std::vector<Eigen::VectorXd> targets = {
		(Eigen::VectorXd(1) << 0).finished(),
		(Eigen::VectorXd(1) << 1).finished(),
		(Eigen::VectorXd(1) << 1).finished(),
		(Eigen::VectorXd(1) << 0).finished(),
	};

	// Training hyperparameters
	int epochs = 10000;
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

	return 0;
}