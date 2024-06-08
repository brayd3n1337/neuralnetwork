#include "NeuralNetwork.h"
#include "../utilities/MathUtil.h"
#include "../propagation/ForwardPropagation.h"
#include "../propagation/BackwardsPropagation.h"
#include "../activation/ActivationUtil.h"
#include <iostream>
#include <random>

NeuralNetwork::NeuralNetwork(const int& inputNodes,
                             const int& hiddenNodes,
                             const int& outputNodes,
                             const double& learningRate,
                             const ActivationType& activationType)
{
    this->inputNodes = inputNodes;
    this->hiddenNodes = hiddenNodes;
    this->outputNodes = outputNodes;
    this->learningRate = learningRate;
    this->activationType = activationType;

    // Initialize weight matrices
    InitializeWeights(weightsInputHidden, hiddenNodes, inputNodes);
    InitializeWeights(weightsHiddenOutput, outputNodes, hiddenNodes);
}

void NeuralNetwork::InitializeWeights(std::vector<std::vector<double>>& weights,
                                      const int& rows,
                                      const int& cols)
{

    // Resize the weights matrix to the specified number of rows and columns
    weights.resize(rows, std::vector<double>(cols));

    std::default_random_engine generator {};
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            weights[i][j] = distribution(generator);
        }
    }
}



void NeuralNetwork::Train(const std::vector<double>& inputs, const std::vector<double>& targets)
{

    ForwardPropagation forwardPropagation;

    // forward propagation
    const auto hiddenOutputs = forwardPropagation.ForwardPropagate(inputs, weightsInputHidden, weightsHiddenOutput, this->activationType)[0];
    const auto finalOutputs = forwardPropagation.ForwardPropagate(inputs, weightsInputHidden, weightsHiddenOutput, this->activationType)[1];


    // calculate output errors
    std::vector<double> outputErrors(targets.size());

    // for each target calculate final output error (not finished because we have to backwards propagate the error)
    for (int i = 0; i < targets.size(); i++)
    {
        // each output error is the target value minus the final output value
        outputErrors[i] = targets[i] - finalOutputs[i];
    }

    // Backpropagation errors
    std::vector<double> hiddenErrors(hiddenNodes);

    BackwardsPropagation backwardsPropagation;

    backwardsPropagation.BackwardsPropagate(outputErrors, weightsHiddenOutput, hiddenErrors);

    // Update weights
    backwardsPropagation.UpdateNodes(outputErrors, hiddenErrors, hiddenOutputs, finalOutputs, inputs,
                                     weightsInputHidden,weightsHiddenOutput, learningRate, inputNodes,
                                     hiddenNodes, outputNodes, this->activationType);
}

std::vector<double> NeuralNetwork::Predict(const std::vector<double>& inputs)
{

    // multiply the inputs by the weights between the input and hidden layers
    std::vector<double> hiddenInputs = MathUtil::Multiply(inputs, weightsInputHidden);

    // create a vector to store the hidden outputs
    std::vector<double> hiddenOutputs(hiddenInputs.size());

    // for each hidden input apply the sigmoid function
    for (int i = 0; i < hiddenInputs.size(); i++)
    {
        // MathUtil::Sigmoid(hiddenInputs[i]);
        hiddenOutputs[i] = ActivationUtil::GetEquation(hiddenInputs[i], this->activationType);
    }

    // multiply the hidden outputs by the weights between the hidden and output layers
    std::vector<double> finalInputs = MathUtil::Multiply(hiddenOutputs, weightsHiddenOutput);

    // create a vector to store the final outputs
    std::vector<double> finalOutputs(finalInputs.size());

    // for each final input
    for (int i = 0; i < finalInputs.size(); i++)
    {
        // apply the sigmoid function to the final inputs
        finalOutputs[i] = ActivationUtil::GetEquation(finalInputs[i], this->activationType);
    }

    // return the final outputs when done
    return finalOutputs;
}
