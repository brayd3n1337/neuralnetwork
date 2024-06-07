

#include <vector>
#include "ForwardPropagation.h"
#include "../utilities/MathUtil.h"

std::vector<std::vector<double>> ForwardPropagation::ForwardPropagate(const std::vector<double>& inputs,const std::vector<std::vector<double>>& hiddenWeightsInput,const std::vector<std::vector<double>>& hiddenWeightsOutput)
{
    // Forward propagation
    std::vector<double> hiddenInputs = MathUtil::Multiply(inputs, hiddenWeightsInput);
    std::vector<double> hiddenOutputs(hiddenInputs.size());

    for (int i = 0; i < hiddenInputs.size(); i++)
    {
        // apply the sigmoid function to the hidden inputs
        hiddenOutputs[i] = MathUtil::Sigmoid(hiddenInputs[i]);
    }

    // multiply the hidden outputs by the weights between the hidden and output layers
    std::vector<double> finalInputs = MathUtil::Multiply(hiddenOutputs, hiddenWeightsOutput);

    // create a vector to store the final outputs
    std::vector<double> finalOutputs(finalInputs.size());

    for (int i = 0; i < finalInputs.size(); i++)
    {
        // apply the sigmoid function to the final inputs
        finalOutputs[i] = MathUtil::Sigmoid(finalInputs[i]);
    }

    // return the final outputs when done
    return {hiddenOutputs, finalOutputs};
}