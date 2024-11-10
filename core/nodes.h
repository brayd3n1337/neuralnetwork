#pragma once

struct nodes
{

	// hidden layers, used when training the neural network
	int hidden;

	// input "layers" / what you feed into the neural network to train it
	int input;

	// the output "layer" / what the neural network "predicts" 
	int output;
};