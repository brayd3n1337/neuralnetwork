#pragma once
#include "propagable/backwards_propagable.h"
#include "../core/nodes.h"

class backwards_propagation
{
public:
	void propagate( backwards_propagable& propagable );

	// ignore this monstrocity, i will improve it later trust me
	void update( nodes& node, const std::vector<double>& output_errors, std::vector<std::vector<double>>& weights_input_hidden, std::vector<std::vector<double>>& weights_hidden_output, const double& learning_rate,
		const std::vector<double>& hidden_errors,
		const std::vector<double>& hidden_outputs,
		const std::vector<double>& final_outputs, const std::vector<double>& inputs );
};

