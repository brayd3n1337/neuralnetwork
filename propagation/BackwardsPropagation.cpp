

#include <vector>
#include "BackwardsPropagation.h"
#include "../activation/ActivationType.h"
#include "../activation/ActivationUtil.h"


void BackwardsPropagation::BackwardsPropagate(const std::vector<double>& errors, const std::vector<std::vector<double>>& weights, std::vector<double>& hiddenErrors)
{
    // foreach hidden error calculate the error
    for (int i = 0; i < hiddenErrors.size(); i++)
    {
        // initialize the error to 0
        auto error = 0.0;

        // for each error
        for (auto j = 0; j < errors.size(); j++)
        {
            // calculate the error
            error += errors[j] * weights[j][i];
        }

        // store the error
        hiddenErrors[i] = error;
    }
}

void BackwardsPropagation::UpdateNodes(const std::vector<double>& outputErrors,
                                const std::vector<double>& hiddenErrors,
                                const std::vector<double>& hiddenOutputs,
                                const std::vector<double>& finalOutputs,
                                const std::vector<double>& inputs,
                                std::vector<std::vector<double>>& weightsInputHidden,
                                std::vector<std::vector<double>>& weightsHiddenOutput,
                                double learningRate,
                                int inputNodes,
                                int hiddenNodes,
                                int outputNodes,
                                ActivationType activationType
                                )
{

    // Update weights between hidden and output layers
    for (int i = 0; i < outputNodes; i++)
    {
        for (int j = 0; j < hiddenNodes; j++)
        {
            weightsHiddenOutput[i][j] += learningRate * outputErrors[i] * ActivationUtil::GetEquation(finalOutputs[i], activationType) * hiddenOutputs[j];
        }
    }

    // Update weights between input and hidden layers
    for (int i = 0; i < hiddenNodes; i++)
    {
        for (int j = 0; j < inputNodes; j++)
        {
            // MathUtil::DSigmoid(hiddenOutputs[i])
            weightsInputHidden[i][j] += learningRate * hiddenErrors[i] * ActivationUtil::GetDerivative(hiddenOutputs[i], activationType) * inputs[j];
        }
    }
}
