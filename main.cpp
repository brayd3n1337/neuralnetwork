#include <iostream>
#include <vector>
#include "NeuralNetwork.h"

int main() {

    // Define the number of nodes in each layer
    const int inputNodes = 3;
    const int hiddenNodes = 3;
    const int outputNodes = 1;

    // Define the learning rate
    const double learningRate = 0.1;

    // Create the neural network
    NeuralNetwork nn(inputNodes, hiddenNodes, outputNodes, learningRate);

    // Example training data (XOR problem)
    const std::vector<std::vector<double>> trainingInputs = {
            {0, 0, 1},
            {0, 1, 1},
            {1, 0, 1},
            {1, 1, 1}
    };

    const std::vector<std::vector<double>> trainingTargets = {
            {0},
            {1},
            {1},
            {0}
    };

    // Define the number of epochs
    int epochs = 100000;

    // Train the neural network for a specified number of epochs (i represents the epoch)
    for (int i = 0; i < epochs; i++)
    {
        for (int j = 0; j < trainingInputs.size(); j++)
        {
            nn.Train(trainingInputs[j], trainingTargets[j]);
        }

        // Print the progress every 1000 epochs
        if ((i + 1) % 1000 == 0)
        {
            std::cout << "Epoch " << (i + 1) << " complete" << std::endl;
        }
    }


    const std::vector<std::vector<double>> testInputs = {
            {0, 0, 1},
            {0, 1, 1},
            {1, 0, 1},
            {1, 1, 1}
    };

    // dogshit code ill recode this later :3
    for (const auto& testInput : testInputs) {
        
        // Predict the output values
        const std::vector<double> result = nn.Predict(testInput);

        // Print the input and output values
        std::cout << "Input: ";

        for (const double val : testInput)
        {
            // Print the input values
            std::cout << val << " ";
        }

        std::cout << "=> Output: ";

        for (const double val : result)
        {
            // Print the output values
            std::cout << val << " ";
        }

        // Print a new line
        std::cout << std::endl;
    }

    return 0;
}
