#pragma once
#include "weight.cpp"


//  The weight factory class is responsible for initializing the neccessary weights of our Neural Network
class weight_factory
{

private:
public:
	weight_factory( std::vector<std::vector<double>>& weights,
		const int& rows,
		const int& col );
};

