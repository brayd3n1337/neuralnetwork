

#ifndef NEURALNETWORK_BACKWARDSPROPAGATION_H
#define NEURALNETWORK_BACKWARDSPROPAGATION_H


class BackwardsPropagation {
public:
    void BackwardsPropagateError(const std::vector<double> &errors, const std::vector<std::vector<double>> &weights, std::vector<double> &hiddenErrors);
};


#endif //NEURALNETWORK_BACKWARDSPROPAGATION_H
