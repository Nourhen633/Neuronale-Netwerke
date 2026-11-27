#include <activation/ScaledSigmoid.h>
#include <cmath>

namespace dnn {
	ScaledSigmoid::ScaledSigmoid(double min2, double max2)
		: min(min2), max(max2) { }

	double ScaledSigmoid::activation(double x) {
		return min + (max - min) * (1 / (1 + std::exp(-x)));
	}

	double ScaledSigmoid::activationBackwards(double x) {
		return (max - min) * (1 / (1 + std::exp(-x))) * (1 - (1 / (1 + std::exp(-x))));
	}

	std::string ScaledSigmoid::getName() const {
		return "scaled_sigmoid";
	}

	double ScaledSigmoid::getMax() const {
		return max;
	}

	double ScaledSigmoid::getMin() const {
		return min;
	}
}
