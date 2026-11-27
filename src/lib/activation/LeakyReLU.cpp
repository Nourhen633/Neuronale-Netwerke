#include <activation/LeakyReLU.h>

namespace dnn {

	double LeakyReLU::activation(double x) {
		return (x > 0) ? x : x * 0.01;
	}

	double LeakyReLU::activationBackwards(double x) {
		return (x > 0) ? 1.0 : 0.01;
	}

	std::string LeakyReLU::getName() const {
		return "leaky_relu";
	}


}
