#include <Network.h>

#include <iostream>
#include <stdexcept>

namespace dnn {

Network::Network(std::vector<std::tuple<std::vector<std::vector<double>>, std::vector<double>, ActivationFunction*>> layerParameters,
            LossFunction*  lossFunction){
    // die mittels lossFunction übergebene Funktion in ownedLossFunction sichern
    ownedLossFunction= lossFunction; 
    
    for (int i=0; i<layerParameters.size();i++){
        // Parametern von Layers erzeugen
        auto weights= std::get<0>(layerParameters[i]);
        auto bias= std::get<1>(layerParameters[i]);
        ActivationFunction* activationFunction= std::get<2>(layerParameters[i]);

        uint64_t layerInputSize= weights[0].size();
        uint64_t layerOutputSize= weights.size();

        //Layers im Netzwerk registrieren
        Layer* layer = new Layer(i,nextSeed++,layerInputSize,layerOutputSize, activationFunction,"Layer"+std::to_string(i));
        layer->setParameters(weights, bias);
        layers.push_back(layer);

    }            
}
Network::~Network(){
    for (Layer* layer : layers) {
        delete layer;
    }
    delete ownedLossFunction;
}
   
Network& addLayer(uint64_t inputDim, uint64_t outputDim, ActivationFunction* activationFunction){
    uint64_t id= layers.size();
    Layer* layer = new Layer(id,nextSeed,inputDim,outputDim, activationFunction,"Layer"+std::to_string(id));
    //initializw weights
    layer->init();
    layers.push_back(layer);
    nextSeed++;
    return *this;
}

const std::vector<double>& getLayerInput(uint64_t layerId) const{
    if(layarID==0){
        return initialInput;
    }
    return layers[layerID-1]->getActivationOutput();
}

void fullForwardPropagation(const std::vector<double>& input){
    initialInput=input;
    std::vector<double> newInput= input;
    for (Layer* layer : layers){
        layer->forwardPropagation(newInput);
        newInput=layer->getActivationOutput();
    }
    output=newInput;
}

void fullBackwardPropagation(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput){
    if(ownedLossFunction==nullptr) {
        throw std::logic_error( "No loss function has been set!" );
    }
    std::vector<double> dACurr =ownedLossFunction-> getGradient(predictedOutput, actualOutput);
    
    for(std::size_t i=layers.size(), i>0;i--){
        Layer* layer= layers[i-1];
        std::vector<double>& LayerInput=getLayerInput(i-1);
        layer-> backwardPropagation(dACurr,layerInput);
        dACurr = layer->getDPreviousLayerOutput();
    }
}

void trainBatch(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& actualOutputs, std::size_t batchSize,
    int epochs, double learningRate){
    std::size_t s=inputs.size();
    if(ownedLossFunction==nullptr) {
        throw std::logic_error( "No loss function has been set!" );
    }
    if (inputs.empty()||actualOutputs.empty()||s!=actualOutputs.size()||batchSize==0||layers.empty()) {
        throw std::invalid_argument(" Invalid inputs or actual outputs or batch size or the network has no layers! ");
    }
    for (const auto& input : inputs) {
        if (input.size()!=layers.front()->getLayerInputSize()) {
            throw std::invalid_argument("Input dimensions do not match the first layer!");
        }
    }

    for (const auto& actualOutput : actualOutputs) {
        if (actualOutput.size()!=layers.back()->getLayerOutputSize()) {
            throw std::invalid_argument("Output dimensions do not match the last layer!");
        }
    }
   
    accuracyHistory.clear();
    costHistory.clear();
    
    for (int epoch= 0; epoch<epochs;epoch++) {
        std::size_t n=inputs.size();
        double epochCost = 0.0;
        double epochAccuracy = 0.0;
        for (std::size_t i = 0; i<n; i++) {
            fullForwardPropagation(inputs[i]);
            epochCost += ownedLossFunction->getCostValue(output,actualOutputs[i]);
            epochAccuracy += ownedLossFunction->getAccuracyValue(output,actualOutputs[i]);
            fullBackwardPropagation(output,actualOutputs[i]);
            if (i % batchSize == 0|| i ==n-1) {
                for (Layer* layer:layers) {
                    layer->commitToWeights(learningRate);
                }
            }
        }
    
        epochCost/=n;
        epochAccuracy/=n;
        costHistory.push_back(epochCost);
        accuracyHistory.push_back(epochAccuracy);
    }    
}

double computeAccuracy(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& actualOutputs){
    std::size_t n=inputs.size();
    if(ownedLossFunction==nullptr) {
        throw std::logic_error( "No loss function has been set!" );
    }
    if (inputs.empty()||actualOutputs.empty()||inputs.size()!=actualOutputs.size()) {
        throw std::invalid_argument(" Invalid inputs or actual outputs! ");
    }
    double totalCorrect=0.0;
    for(std::size_t i=0;i<n;i++){
        fullForwardPropagation(inputs[i]);
        totalCorrect+=ownedLossFunction->getAccuracyValue(output,actualOutputs[i]);
    }
    
    return totalCorrect/n;
}

void Network::setLossFunction(LossFunction* lossFunction) {
    delete this->ownedLossFunction;
    this->ownedLossFunction = lossFunction;
}

const LossFunction* Network::getLossFunction() const {
    return ownedLossFunction;
}

const std::vector<double>& Network::getOutput() const {
    return output;
}

const std::vector<double>& Network::getCostHistory() const {
    return costHistory;
}

const std::vector<double>& Network::getAccuracyHistory() const {
    return accuracyHistory;
}

const std::vector<Layer*>& Network::getLayers() const {
    return layers;
}

void Network::setLayers(std::vector<Layer*> newLayers) {
    for (Layer* layer : layers) {
        delete layer;
    }
    layers = std::move(newLayers);
    initialInput.clear();
    output.clear();
    costHistory.clear();
    accuracyHistory.clear();
}

}