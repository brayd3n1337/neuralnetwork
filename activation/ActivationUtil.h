

#ifndef NEURALNETWORK_ACTIVATIONUTIL_H
#define NEURALNETWORK_ACTIVATIONUTIL_H


#include "ActivationType.h"

class ActivationUtil {
private:
    static double Sigmoid(const double& x);

    static double ReLU(const double& x);

    static double LeakyReLU(const double& x);

    static double DSigmoid(const double& y);

    static double DTanh(const double& y);

    static double DReLU(const double& y);

    static double DLeakyReLU(const double& y);
public:
    static double GetEquation(const double& x, const ActivationType type);

    static double GetDerivative(const double &y, const ActivationType type);
};


#endif //NEURALNETWORK_ACTIVATIONUTIL_H
