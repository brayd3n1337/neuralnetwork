

#include <cmath>
#include <vector>
#include "MathUtil.h"


// Sigmoid function
double MathUtil::Sigmoid(const double& x)
{
    return 1.0 / (1.0 + exp(-x));
}

// finds the derivative of the sigmoid function.
double MathUtil::DSigmoid(const double& y)
{
    return y * (1.0 - y);
}


// Multiply a vector by a matrix
std::vector<double> MathUtil::Multiply(const std::vector<double>& a, const std::vector<std::vector<double>>& b)
{
    std::vector<double> result(b.size(), 0.0);

    for (int i = 0; i < b.size(); i++)
    {
        for (int j = 0; j < a.size(); j++)
        {
            result[i] += a[j] * b[i][j];
        }
    }

    return result;
}