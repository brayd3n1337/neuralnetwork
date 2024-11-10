#pragma once

#include "propagable/forwards_propagable.h"
#include <vector>

class forwards_propagation
{
public:
	std::vector<std::vector<double>> propagate( forwards_propagable& propagable );
};

