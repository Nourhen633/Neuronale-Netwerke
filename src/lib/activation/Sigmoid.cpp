#include <activation/Sigmoid.h>
#include <cmath> //bring e^-x

namespace dnn {
	double Sigmoid::activation(double x) {
		return (1.0 / (1.0 + std::exp(-x)));
	}

	double Sigmoid::activationBackwards(double x) {
		double res = activation(x);
		return res * (1.0 - res);
	}

	std::string Sigmoid::getName() const {
		return "sigmoid";
	}


}
