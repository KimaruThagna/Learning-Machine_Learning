from math import exp
from random import seed
import random

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs): # number of neurons/layer
	network = list()
	hidden_layer = [{'weights':[round(random.uniform(0, 1),3) for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer) # +1 for bias
	output_layer = [{'weights':[round(random.uniform(0, 1),3) for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1] # bias
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i] #sum of products
	return activation

# neuron activation
def transfer(activation): # sigmoid func
	return 1.0 / (1.0 + exp(-activation))


def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs) #SOP
			neuron['output'] = round(transfer(activation),3)# sigmoid
			new_inputs.append(neuron['output'])
		inputs = new_inputs# output of layer 1 is input of layer 2
	return inputs # the output of the output layer

# Calculate the derivative of an neuron output
def transfer_derivative(output):
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1: # not the output layer
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += round((neuron['weights'][j] * neuron['delta']),3)
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = round(errors[j] * transfer_derivative(neuron['output']),3)

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += round(l_rate * neuron['delta'] * inputs[j],3) #nex
			neuron['weights'][-1] += round(l_rate * neuron['delta'],3)

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)

		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
		print(expected)

# Test training backprop algorithm
seed(1)

def predict(network, row):
    outputs = forward_propagate(network, row)
    print(outputs)
    return outputs.index(max(outputs))# class of the larger probability


# Test making predictions with the network
dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [8.675418651, -0.242068655, 1],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.673756466, 3.508563011, 1]]

test_set=[ [7.627531214, 2.759262235, 1],
            [3.396561688, 4.400293529, 0],
           [6.922596716, 1.77106367, 1],]
#training phase
n_inputs = len(dataset[0]) - 1
n_outputs =len(set([row[-1] for row in dataset]))
network = initialize_network(n_inputs, 2, n_outputs)
for layer in network:
    print(str(layer) +'\n')
train_network(network, dataset, 0.996, 10, n_outputs)
for layer in network:
 	print(str(layer)+'\n')

#testing phase
for row in test_set:
    print('Expected=%d, Got=%d' % (row[-1],  predict(network, row)))