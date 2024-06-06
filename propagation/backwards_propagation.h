

#ifndef NEURALNETWORK_BACKWARDS_PROPAGATION_H
#define NEURALNETWORK_BACKWARDS_PROPAGATION_H


class BackwardsPropagation {
public:
    void BackwardsPropagateError(const std::vector<double> &errors, const std::vector<std::vector<double>> &weights, std::vector<double> &hiddenErrors);
};


#endif //NEURALNETWORK_BACKWARDS_PROPAGATION_H
