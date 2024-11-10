#include "forwards_propagation.h"
#include "../math/math_util.h"

std::vector<std::vector<double>> forwards_propagation::propagate( forwards_propagable& propagable )
{
	const std::vector<double> hidden_inputs = math_util::multiply_matrix( propagable.inputs, propagable.hidden_weights_input );
	
	std::vector<double> hidden_outputs( hidden_inputs.size( ) );

	for ( int i = 0; i < hidden_inputs.size( ); i++ )
	{
		hidden_outputs[i] = math_util::sigmoid( hidden_inputs[i] );
	}

	const std::vector<double> inputs = math_util::multiply_matrix( hidden_outputs, propagable.hidden_weights_output );

	std::vector<double> outputs( inputs.size( ) );

	for ( int i = 0; i < inputs.size( ); i++ )
	{
		outputs[i] = math_util::sigmoid( inputs[i] );
	}

	return { hidden_outputs, outputs };
}