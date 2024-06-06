#ifndef NEURALNETWORK_NEURAL_NETWORK_H
#define NEURALNETWORK_NEURAL_NETWORK_H

#include <vector>
#include <cstdlib>
#include <cmath>

class NeuralNetwork {
private:
    int inputNodes, hiddenNodes, outputNodes;
    double learningRate;

    std::vector<std::vector<double>> weightsInputHidden;
    std::vector<std::vector<double>> weightsHiddenOutput;

public:
    NeuralNetwork(const int& inputNodes,
                  const int& hiddenNodes,
                  const int& outputNodes,
                  const double& learningRate);

    void InitializeWeights(std::vector<std::vector<double>>& weights,
                           const int& rows,
                           const int& cols);

    void Train(const std::vector<double>& inputs,
               const std::vector<double>& targets);

    std::vector<double> Predict(const std::vector<double>& inputs);

    void UpdateNodes(const std::vector<double>& outputErrors,
                     const std::vector<double>& hiddenErrors,
                     const std::vector<double>& hiddenOutputs,
                     const std::vector<double>& finalOutputs,
                     const std::vector<double>& inputs);
};

#endif //NEURALNETWORK_NEURAL_NETWORK_H
