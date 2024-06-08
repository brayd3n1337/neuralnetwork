

#include <valarray>
#include "ActivationUtil.h"

double ActivationUtil::GetEquation(const double& x, const ActivationType type)
{
    switch (type)
    {
        case ActivationType::Sigmoid:
            return Sigmoid(x);
        case ActivationType::ReLU:
            return ReLU(x);
        case ActivationType::LeakyReLU:
            return LeakyReLU(x);
        case ActivationType::Tanh:
            return tanh(x);
        default:
            return 0;
    }
}

double ActivationUtil::LeakyReLU(const double &x)
{
    return x > 0 ? x : 0.01 * x;
}

double ActivationUtil::Sigmoid(const double& x)
{
    return 1.0 / (1.0 + exp(-x));
}

double ActivationUtil::ReLU(const double &x)
{
    return x > 0 ? x : 0;
}

double ActivationUtil::GetDerivative(const double& y, const ActivationType type)
{
    switch (type)
    {
        case ActivationType::Sigmoid:
            return y * (1.0 - y);
        case ActivationType::ReLU:
            return DReLU(y);
        case ActivationType::LeakyReLU:
            return DLeakyReLU(y);
        case ActivationType::Tanh:
            return DTanh(y);
        default:
            return 0;
    }
}

double ActivationUtil::DReLU(const double &y)
{
    return y > 0 ? 1 : 0;
}

double ActivationUtil::DLeakyReLU(const double &y)
{
    return y > 0 ? 1 : 0.01;
}

double ActivationUtil::DTanh(const double &y)
{
    return 1.0 - y * y;
}

double ActivationUtil::DSigmoid(const double& y)
{
    return y * (1.0 - y);
}
