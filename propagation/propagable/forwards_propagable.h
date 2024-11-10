#pragma once

#include <vector>

struct forwards_propagable
{
    std::vector<double>& inputs;
    std::vector<std::vector<double>>& hidden_weights_input;
    std::vector<std::vector<double>>& hidden_weights_output;
};