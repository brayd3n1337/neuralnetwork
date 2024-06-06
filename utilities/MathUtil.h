

#ifndef NEURALNETWORK_MATHUTIL_H
#define NEURALNETWORK_MATHUTIL_H


class MathUtil {
public:
    static double Sigmoid(const double& x);

    static double DSigmoid(const double& y) ;

    static std::vector<double> Multiply(const std::vector<double> &a, const std::vector<std::vector<double>> &b);
};


#endif //NEURALNETWORK_MATHUTIL_H
