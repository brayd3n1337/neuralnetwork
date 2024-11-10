#pragma once

#include <vector>

struct backwards_propagable
{
	std::vector<double>& errors;
	std::vector<std::vector<double>>& weights;
	std::vector<double>& hidden_errors;
};