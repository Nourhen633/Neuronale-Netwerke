#pragma once

#include <activation/ActivationFunction.h>

#include <string>

namespace dnn {

    class Sigmoid : public ActivationFunction {
    public:
        Sigmoid() = default;
        ~Sigmoid() = default;

        virtual double activation(double x) override;
        virtual double activationBackwards(double x) override;
        virtual std::string getName() const override;
    };

}
