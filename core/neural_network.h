#pragma once

#include "nodes.h"
#include <vector>

class neural_network
{
private:
	nodes m_node;
	double m_learning_rate;
public:
	neural_network( nodes& node, const double& learning_rate);

	void train( std::vector<double>& inputs, const std::vector<double>& targets );

	std::vector<double> predict( const std::vector<double>& inputs );
};