cmake_minimum_required(VERSION 3.28)
project(neuralnetwork)

set(CMAKE_CXX_STANDARD 17)

add_executable(neuralnetwork main.cpp
        NeuralNetwork.cpp
        NeuralNetwork.h
        utilities/MathUtil.cpp
        utilities/MathUtil.h
        propagation/ForwardPropagation.cpp
        propagation/ForwardPropagation.h
        propagation/BackwardsPropagation.cpp
        propagation/BackwardsPropagation.h
)
