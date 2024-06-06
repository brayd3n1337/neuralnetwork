#include "neural_network.h"
#include "utilities/math_util.h"
#include "propagation/forward_propagation.h"
#include "propagation/backwards_propagation.h"
#include <iostream>
#include <random>

NeuralNetwork::NeuralNetwork(const int& inputNodes, const int& hiddenNodes, const int& outputNodes, const double& learningRate) {
    this->inputNodes = inputNodes;
    this->hiddenNodes = hiddenNodes;
    this->outputNodes = outputNodes;
    this->learningRate = learningRate;

    // Initialize weight matrices
    InitializeWeights(weightsInputHidden, hiddenNodes, inputNodes);
    InitializeWeights(weightsHiddenOutput, outputNodes, hiddenNodes);
}

void NeuralNetwork::InitializeWeights(std::vector<std::vector<double>>& weights, const int& rows, const int& cols)
{

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
    auto hiddenOutputs = forwardPropagation.ForwardPropagate(inputs, weightsInputHidden, weightsHiddenOutput)[0];
    auto finalOutputs = forwardPropagation.ForwardPropagate(inputs, weightsInputHidden, weightsHiddenOutput)[1];


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

    backwardsPropagation.BackwardsPropagateError(outputErrors, weightsHiddenOutput, hiddenErrors);

    // Update weights
    this->UpdateNodes(outputErrors, hiddenErrors, hiddenOutputs, finalOutputs, inputs);
}

void NeuralNetwork::UpdateNodes(const std::vector<double>& outputErrors,
                                const std::vector<double>& hiddenErrors,
                                const std::vector<double>& hiddenOutputs,
                                const std::vector<double>& finalOutputs,
                                const std::vector<double>& inputs)
{

    // Update weights between hidden and output layers
    for (int i = 0; i < outputNodes; i++)
    {
        for (int j = 0; j < hiddenNodes; j++)
        {
            // this math was used from gpt
            weightsHiddenOutput[i][j] += learningRate * outputErrors[i] * MathUtil::DSigmoid(finalOutputs[i]) * hiddenOutputs[j];
        }
    }

    // Update weights between input and hidden layers
    for (int i = 0; i < hiddenNodes; i++)
    {
        for (int j = 0; j < inputNodes; j++)
        {
            // this math was used from gpt
            weightsInputHidden[i][j] += learningRate * hiddenErrors[i] * MathUtil::DSigmoid(hiddenOutputs[i]) * inputs[j];
        }
    }
}

std::vector<double> NeuralNetwork::Predict(const std::vector<double>& inputs)
{

    // multiply the inputs by the weights between the input and hidden layers
    std::vector<double> hiddenInputs = MathUtil::Multiply(inputs, weightsInputHidden);

    // create a vector to store the hidden outputs
    std::vector<double> hiddenOutputs(hiddenInputs.size());

    // for each hidden input
    for (int i = 0; i < hiddenInputs.size(); i++)
    {
        // apply the sigmoid function to the hidden inputs
        hiddenOutputs[i] = MathUtil::Sigmoid(hiddenInputs[i]);
    }

    // multiply the hidden outputs by the weights between the hidden and output layers
    std::vector<double> finalInputs = MathUtil::Multiply(hiddenOutputs, weightsHiddenOutput);

    // create a vector to store the final outputs
    std::vector<double> finalOutputs(finalInputs.size());

    // for each final input
    for (int i = 0; i < finalInputs.size(); i++)
    {
        // apply the sigmoid function to the final inputs
        finalOutputs[i] = MathUtil::Sigmoid(finalInputs[i]);
    }

    // return the final outputs when done
    return finalOutputs;
}
