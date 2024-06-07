

#include <vector>
#include "BackwardsPropagation.h"

void BackwardsPropagation::BackwardsPropagateError(const std::vector<double>& errors, const std::vector<std::vector<double>>& weights, std::vector<double>& hiddenErrors)
{
    // foreach hidden error calculate the error
    for (int i = 0; i < hiddenErrors.size(); i++)
    {
        // initialize the error to 0
        auto error = 0.0;

        // for each error
        for (auto j = 0; j < errors.size(); j++)
        {
            // calculate the error
            error += errors[j] * weights[j][i];
        }

        // store the error
        hiddenErrors[i] = error;
    }
}