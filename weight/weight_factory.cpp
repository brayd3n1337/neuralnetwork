#include "weight_factory.h"
#include <random>

weight_factory::weight_factory( std::vector<std::vector<double>>& weights_matrix,
	const int& rows,
	const int& columns )
{
	// resize the weight matrix to fit the rows and columns
    weights_matrix.resize( rows, std::vector<double>( columns ) );

    std::default_random_engine generator {};
    std::uniform_real_distribution<double> distribution( -1.0, 1.0 );

    for ( int i = 0; i < rows; i++ )
    {
        for ( int j = 0; j < columns; j++ )
        {
            weights_matrix[i][j] = distribution( generator );
        }
    }
}