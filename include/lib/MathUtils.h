#pragma once

#include <vector>
#include <algorithm>
#include <stdexcept>

namespace dnn {

double dotProduct(const std::vector<double>& a, const std::vector<double>& b);

double sum(const std::vector<double>& values);

// Encodes value as a one-hot vector of numClasses elements: the element at index value is 1.0
// and every other element is 0.0. Throws std::out_of_range if value is negative or >= numClasses.
std::vector<double> encodeAsOneHot(double value, std::size_t numClasses);


template <typename T>
std::vector<double> normalize(const std::vector<T>& values) {

	if (values.empty()) return {};
	T max = values[0];
	for (std::size_t i = 1; i < values.size(); i++) {
		if (values[i] > max) max = values[i];
	}

	const int maxVal = static_cast<int>(max);

	std::vector<double> ready(values.size());
	for (std::size_t i = 0; i < values.size(); i++) {
		ready[i] = (double) values[i] / maxVal;
		
	}
	return ready;

}

template <typename T>
std::vector<double> normalize(const std::vector<T>& values, int maxValue) {

	if (maxValue == 0) throw std::invalid_argument("normalize: max is zero");

	std::vector<double> ready(values.size());
	for (std::size_t i = 0; i < values.size(); i++) {
		ready[i] = (double)values[i] / maxValue;
		
	}
	return ready;


}

}
