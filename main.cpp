#include <Eigen/Dense>
#include <iostream>
using namespace Eigen;
using namespace std;

int main() {
	// Dynamic matrix - resizable
	MatrixXd d;

	// Fixed sizes matrix
	Matrix3d f;

	

	d = MatrixXd::Random(4,6);

	cout << d << endl;
}