#pragma once

#include <activation/ActivationFunction.h>

#include <string>

namespace dnn {
	class ReLU : public ActivationFunction {
	public:
		ReLU() = default;
		~ReLU() = default;

		virtual double activation(double x) override;
		virtual double activationBackwards(double x) override;
		virtual std::string getName() const override;

	};
}
