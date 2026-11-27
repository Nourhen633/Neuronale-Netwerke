#pragma once

#include <activation/ActivationFunction.h>

#include <string>

namespace dnn {

    // Sigmoid whose output is linearly rescaled from (0, 1) to (min, max).
    class ScaledSigmoid : public ActivationFunction {
    public:
        ScaledSigmoid(double min, double max);
        ~ScaledSigmoid() = default;

        double getMin() const;
        double getMax() const;

        virtual double activation(double x) override;
        virtual double activationBackwards(double x) override;
        virtual std::string getName() const override;

    private:
        double min;
        double max;
    };

}
