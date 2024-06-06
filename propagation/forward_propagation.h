

#ifndef NEURALNETWORK_FORWARD_PROPAGATION_H
#define NEURALNETWORK_FORWARD_PROPAGATION_H


/**
 *
 * Process of passing in the input data and calculating the output of the neural network
 *
 * Input Layer -> Hidden Layer -> Output Layer
 *
 * Input data is passed into the input layer, which is then multiplied by the weights connecting the input layer to the hidden layer.
 * The result is then passed through an activation function (sigmoid function) to produce the output of the hidden layer.
 *
 * I would like to add other algorithms besides sigmoid but im lazy and don't know anything about them
 */
class ForwardPropagation {
public:
    std::vector<std::vector<double>> ForwardPropagate(const std::vector<double> &inputs, const std::vector<std::vector<double>> &hiddenWeightsInput, const std::vector<std::vector<double>> &hiddenWeightsOutput);
};


#endif //NEURALNETWORK_FORWARD_PROPAGATION_H
