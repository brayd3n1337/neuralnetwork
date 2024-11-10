#include "backwards_propagation.h"
#include "../math/math_util.h"

void backwards_propagation::propagate( backwards_propagable& propagable )
{
    for ( int i = 0; i < propagable.hidden_errors.size( ); i++ )
    {
        // initialize the error to 0
        auto error = 0.0;

        // for each error
        for ( auto j = 0; j < propagable.errors.size( ); j++ )
        {
            // calculate the error
            error += propagable.errors[j] * propagable.weights[j][i];
        }

        // assign the hidden errors
        propagable.hidden_errors[i] = error;
    }
}

void backwards_propagation::update( nodes& node, const std::vector<double>& output_errors, std::vector<std::vector<double>>& weights_input_hidden, std::vector<std::vector<double>>& weights_hidden_output, const double& learning_rate,
    const std::vector<double>& hidden_errors, 
    const std::vector<double>& hidden_outputs,
    const std::vector<double>& final_outputs, const std::vector<double>& inputs )
{
    for ( int i = 0; i < node.output; i++ )
    {
        for ( int j = 0; j < node.hidden; j++ )
        {
            weights_hidden_output[i][j] += learning_rate * output_errors[i] * math_util::sigmoid( final_outputs[i] ) * hidden_outputs[j];
        }
    }

    // Update weights between input and hidden layers
    for ( int i = 0; i < node.hidden; i++ )
    {
        for ( int j = 0; j < node.input; j++ )
        {
            // MathUtil::DSigmoid(hiddenOutputs[i])
            weights_input_hidden[i][j] += learning_rate * hidden_errors[i] * math_util::derivative_sigmoid( hidden_outputs[i] ) * inputs[j];
        }
    }
}