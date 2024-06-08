#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H

#include <vector>
#include <cstdlib>
#include <cmath>
#include "../activation/ActivationType.h"

class NeuralNetwork {
private:
    int inputNodes, hiddenNodes, outputNodes;
    double learningRate;

    ActivationType activationType;

    std::vector<std::vector<double>> weightsInputHidden;
    std::vector<std::vector<double>> weightsHiddenOutput;

public:
    NeuralNetwork(const int& inputNodes,
                  const int& hiddenNodes,
                  const int& outputNodes,
                  const double& learningRate,
                  const ActivationType& activationType);

    void InitializeWeights(std::vector<std::vector<double>>& weights,
                           const int& rows,
                           const int& cols);

    void Train(const std::vector<double>& inputs, const std::vector<double>& targets);

    std::vector<double> Predict(const std::vector<double> &inputs);
};

#endif //NEURALNETWORK_NEURALNETWORK_H
