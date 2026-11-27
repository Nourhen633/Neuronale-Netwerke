#pragma once

#include <activation/ActivationFunction.h>

#include <string>

namespace dnn{

class LeakyReLU : public ActivationFunction {
public:
    LeakyReLU() = default;
    ~LeakyReLU() = default;

    virtual double activation(double x) override;
    virtual double activationBackwards(double x) override;
    virtual std::string getName() const override;
};
}

