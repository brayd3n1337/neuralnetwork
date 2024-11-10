#include <iostream>
#include "core/nodes.h"
#include "core/neural_network.h"
#include <memory>

int main( )
{
    const int input_nodes = 3;
    const int hidden_nodes = 3;
    const int output_nodes = 1;

    nodes node = { hidden_nodes, input_nodes, output_nodes };
    const double learning_rate = 0.1;

    

    std::unique_ptr<neural_network> nn = std::make_unique<neural_network>( node, learning_rate );

    std::vector<std::vector<double>> inputs =
    {
        {0, 0, 1},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 1}
    };

    const std::vector<std::vector<double>> targets =
    {
        {0},
        {1},
        {1},
        {0}
    };

    const int epochs = 100000;

    for ( int i = 0; i < epochs; i++ )
    {
        for ( int j = 0; j < inputs.size( ); j++ )
        {
            nn->train( inputs[j], targets[j] );
        }

        // for each 1000 epoch write epoch done
        if ( ( i + 1 ) % 1000 == 0 )
        {
            std::cout << "epoch " << ( i + 1 ) << " complete" << std::endl;
        }
    }

    const std::vector<std::vector<double>> testInputs = {
        {0, 0, 1},
        {0, 1, 1},
        {1, 0, 1},
        {1, 1, 1}
    };

    for ( const auto& testInput : testInputs ) {
        const std::vector<double> result = nn->predict( testInput );

        std::cout << "Input: ";

        for ( const double val : testInput )
        {
            std::cout << val << " ";
        }

        std::cout << "=> Output: ";

        for ( const double& val : result )
        {
            std::cout << val << " ";
        }

        std::cout << std::endl;
    }


}
