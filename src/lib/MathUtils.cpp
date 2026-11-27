#include <MathUtils.h>

#include <stdexcept>
#include <string>

namespace dnn {

    double dotProduct(const std::vector<double>& a, const std::vector<double>& b) {

        if (a.size() != b.size()) return 0.0; // we cannot compute a scalarproduct if they are not equal

        std::vector<double> values(a.size());
        for (std::size_t i = 0; i < a.size(); i++) {
            values[i] = a[i] * b[i];
        }
        return sum(values);
    }
    
    double sum(const std::vector<double>& values) {
        double summe = 0.0;
        //size=3; 01 12 23
        for (std::size_t i = 0; i < values.size(); i++) {
            summe += values[i];
        }
        return summe;
  
    }

std::vector<double> encodeAsOneHot(double value, std::size_t numClasses) {
    if (value < 0.0 || value >= static_cast<double>(numClasses)) {
        throw std::out_of_range("encodeAsOneHot: value " + std::to_string(value) + " is out of range for " +
                                std::to_string(numClasses) + " classes");
    }

    std::vector<double> encoded(numClasses, 0.0);
    encoded[static_cast<std::size_t>(value)] = 1.0;
    return encoded;
}

}
