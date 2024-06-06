cmake_minimum_required(VERSION 3.28)
project(neuralnetwork)

set(CMAKE_CXX_STANDARD 17)

add_executable(neuralnetwork main.cpp
        neural_network.cpp
        neural_network.h
        utilities/math_util.cpp
        utilities/math_util.h
        propagation/forward_propagation.cpp
        propagation/forward_propagation.h
        propagation/backwards_propagation.cpp
        propagation/backwards_propagation.h
)
