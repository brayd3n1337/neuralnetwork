#include "neural_network.h"
#include "../weight/weight_factory.h"
#include "../propagation/forwards_propagation.h"
#include "../propagation/backwards_propagation.h"
#include "../math/math_util.h"
#include <memory>

std::vector<std::vector<double>> m_weight_inputs_hidden;
std::vector<std::vector<double>> m_weight_outputs_hidden;

neural_network::neural_network( nodes& node, const double& learning_rate )
{ 
	this->m_node = node;
	this->m_learning_rate = learning_rate;
	
	// initialize the weights
	const weight_factory _( m_weight_inputs_hidden, node.hidden, node.input );
	const weight_factory __( m_weight_outputs_hidden, node.output, node.hidden );
}

void neural_network::train( std::vector<double>& inputs, const std::vector<double>& targets )
{

	std::unique_ptr<forwards_propagation> forwards = std::make_unique<forwards_propagation>( );
	std::unique_ptr<backwards_propagation> backwards = std::make_unique<backwards_propagation>( );

	forwards_propagable forwards_prop = { inputs , m_weight_inputs_hidden, m_weight_outputs_hidden };


	const auto hidden_outputs = forwards->propagate( forwards_prop )[0];
	const auto outputs = forwards->propagate( forwards_prop )[1];

	std::vector<double> errors( targets.size( ) );

	for ( int i = 0; i < targets.size( ); i++ )
	{
		errors[i] = targets[i] - outputs[i];
	}

	std::vector<double> hidden_errors( this->m_node.hidden );

	backwards_propagable backwards_prop = { errors, m_weight_outputs_hidden, hidden_errors };

	backwards->propagate( backwards_prop );

	backwards->update( this->m_node, errors, m_weight_inputs_hidden, m_weight_outputs_hidden, this->m_learning_rate, hidden_errors, hidden_outputs, outputs, inputs );
}

std::vector<double> neural_network::predict( const std::vector<double>& inputs )
{
	std::vector<double> hidden_inputs = math_util::multiply_matrix( inputs, m_weight_inputs_hidden );

	std::vector<double> hidden_outputs( hidden_inputs.size( ) );

	for ( int i = 0; i < hidden_inputs.size( ); i++ )
	{
		hidden_outputs[i] = math_util::sigmoid( hidden_inputs[i]);
	}

	std::vector<double> final_inputs = math_util::multiply_matrix( hidden_outputs, m_weight_outputs_hidden );

	std::vector<double> outputs( inputs.size( ) );

	for ( int i = 0; i < final_inputs.size( ); i++ )
	{
		outputs[i] = math_util::sigmoid( final_inputs[i] );
	}

	// return the final outputs when done
	return outputs;
}