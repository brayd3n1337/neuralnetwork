#pragma once

#include <cmath>
#include "vector"

class math_util
{
private:
	// Uninitializable constructor.
	math_util( );
public:

	static const double sigmoid( const double& x )
	{
		return 1.0 / ( 1.0 + exp( -x ) );
	}

	static const double derivative_sigmoid( const double& y )
	{
		return y * ( 1.0 - y );
	}

	// https://github.com/brayd3n1337/neuralnetwork/blob/main/utilities/MathUtil.cpp
	// multiplies a vector by a matrix
	static const std::vector<double> multiply_matrix( const std::vector<double>& a, const std::vector<std::vector<double>>& b )
	{
		std::vector<double> result( b.size( ), 0.0 );

		for ( int i = 0; i < b.size( ); i++ )
		{
			for ( int j = 0; j < a.size( ); j++ )
			{
				result[i] += a[j] * b[i][j];
			}
		}

		return result;
	}


};

