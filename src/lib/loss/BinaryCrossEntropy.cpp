#include <loss/BinaryCrossEntropy.h>

#include <MathUtils.h>

#include <algorithm>
#include <cmath>

namespace dnn {

std::string BinaryCrossEntropy::getName() const {
    return "binary_cross_entropy";
}

double BinaryCrossEntropy::getCostValue(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const{
    std::size_t n =predictedOutput.size();
    std::vector<double> P;
    std::vector<double> Q;
    std::vector<double> R;
    for(std::size_t i=0;i<n;i++){
        double a=std::log(clampProbability(predictedOutput[i]));
        double b=std::log(1-clampProbability(predictedOutput[i]));
        double c=1-actualOutput[i];
        P.push_back(a);
        Q.push_back(b);
        R.push_back(c);
    }
    return -(dotProduct(actualOutput,P)+dotProduct(R,Q))/n;
}

double getAccuracyValue(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const {
    std::size_t n =predictedOutput.size();
    std::size_t correct=0;
    std::vector<double> V;
    for(std::size_t i=0;i<n;i++){
        double v;
        if(predictedOutput[i]>0.5){
            v=1.0;
        }else{
            v=0.0;
        }
        V.push_back(v);
        if (V[i] == actualOutput[i]) {
            correct++;
        }
    }
    //to ensure floating-point division
    return static_cast<double>(correct)/n;
}

std::vector<double> getGradient(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const{
    std::vector<double> G;
    std::size_t n =predictedOutput.size();
    for(std::size_t i=0;i<n;i++){
        double g=(-1.0/n)*((actualOutput[i]/clampProbability(predictedOutput[i])) - ((1-actualOutput[i])/(1-clampProbability(predictedOutput[i]))));
        G.push_back(g);
    }
    return G;
}
}
