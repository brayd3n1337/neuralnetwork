

#ifndef NEURALNETWORK_BACKWARDSPROPAGATION_H
#define NEURALNETWORK_BACKWARDSPROPAGATION_H


#include "../activation/ActivationType.h"

class BackwardsPropagation {
public:
    void BackwardsPropagate(const std::vector<double> &errors, const std::vector<std::vector<double>> &weights, std::vector<double> &hiddenErrors);



    void UpdateNodes(const std::vector<double> &outputErrors, const std::vector<double> &hiddenErrors,
                     const std::vector<double> &hiddenOutputs, const std::vector<double> &finalOutputs,
                     const std::vector<double> &inputs, std::vector<std::vector<double>> &weightsInputHidden,
                     std::vector<std::vector<double>> &weightsHiddenOutput, double learningRate, int inputNodes,
                     int hiddenNodes, int outputNodes, ActivationType activationType);
};


#endif //NEURALNETWORK_BACKWARDSPROPAGATION_H
