#include <iostream>
#include <cmath>

#include "MLP.h"

#define RANDOM_WEIGHT (double)((std::rand() % 2400) - 1200) / 1000.0

NeuralNetwork::MLP::MLP(unsigned int nb_layers) {
  std::vector<double>tmpvd;
  std::vector< std::vector<double> >tmpvvd;

  if (nb_layers > 1) {
    for (unsigned int layer = 0; layer < nb_layers; ++layer) {
      m_activation.push_back(tmpvd);
      m_weights.push_back(tmpvvd);
      m_deltaWeights.push_back(tmpvvd);
      m_sum.push_back(tmpvd);
      m_delta.push_back(tmpvd);
      m_sumDeltaWeights.push_back(tmpvd);
    }
    for (unsigned int layer = 0; layer < nb_layers; ++layer)
      addNeuron(layer);//adding bias neurons
  } else throw ConstructionException();
}

void NeuralNetwork::MLP::addNeuron(unsigned int layer) {
  std::vector<double>tmpvd;

  m_activation[layer].push_back(1.0);
  m_sum[layer].push_back(0.0);
  m_delta[layer].push_back(0.0);
  m_sumDeltaWeights[layer].push_back(0.0);

  if (layer < m_activation.size() - 1) {
    m_weights[layer].push_back(tmpvd);
    for (unsigned int nb_neurons = m_activation[layer+1].size(); nb_neurons > 0; --nb_neurons)
      m_weights[layer][m_activation[layer].size() - 1].push_back(RANDOM_WEIGHT);
    m_deltaWeights[layer].push_back(tmpvd);
    for (unsigned int nb_neurons = m_activation[layer+1].size(); nb_neurons > 0; --nb_neurons)
      m_deltaWeights[layer][m_activation[layer].size() - 1].push_back(0.0);
  }

  if (layer > 0) {
    for (unsigned int nb_neurons = m_activation[layer-1].size(); nb_neurons > 0; --nb_neurons)
      m_weights[layer-1][nb_neurons-1].push_back(RANDOM_WEIGHT);
    for (unsigned int nb_neurons = m_activation[layer-1].size(); nb_neurons > 0; --nb_neurons)
      m_deltaWeights[layer-1][nb_neurons-1].push_back(0.0);
  }
}

void NeuralNetwork::MLP::process(const std::vector<double>& input, vector<double>& output) {
  unsigned int outputLayer = m_activation.size()-1;

  if (input.size() != m_activation[0].size() - 1 ||
      output.size() != m_activation[outputLayer].size() - 1) {
    throw ArgumentException();
    return;
  }
  for (unsigned int neuronIn = 1, neuronInMax = m_activation[0].size();
       neuronIn < neuronInMax; ++neuronIn)
    m_activation[0][neuronIn] = input[neuronIn-1];


  for (unsigned int layer = 1, layerMax = m_activation.size(); layer < layerMax; ++layer) {
    for (unsigned int neuronOut = 1, neuronOutMax = m_activation[layer].size();
	 neuronOut < neuronOutMax;
	 ++neuronOut) {
      m_sum[layer][neuronOut] = m_weights[layer-1][0][neuronOut];
      for (unsigned int neuronIn = 0, neuronInMax = m_activation[layer-1].size();
	   neuronIn < neuronInMax;
	   ++neuronIn) {
	m_sum[layer][neuronOut] += m_activation[layer-1][neuronIn]*m_weights[layer-1][neuronIn][neuronOut];
      }
      m_activation[layer][neuronOut] = 1.0 / (1.0 + std::exp(-m_sum[layer][neuronOut]));
    }
  }
  for (int neuronOut = 1, neuronOutMax = m_activation[outputLayer].size();
       neuronOut < neuronOutMax; ++neuronOut) output[neuronOut-1] = m_activation[outputLayer][neuronOut];
  return;
}

double NeuralNetwork::MLP::train(const std::vector<double>& input,
				 std::vector<double>& output,
				 const std::vector<double>& target) {
  double error = 0.0;
  unsigned int outputLayer = m_activation.size()-1;

  if (input.size() != m_activation[0].size() - 1 ||
      output.size() != m_activation[outputLayer].size() - 1 ||
      target.size() != m_activation[outputLayer].size() - 1) {
    throw ArgumentException();
    return error;
  }

  process(input, output);
  
  //error calculation
  for (unsigned int neuronOut = 1, neuronOutMax = m_activation[outputLayer].size();
       neuronOut < neuronOutMax;
       ++neuronOut) {
    error += (target[neuronOut-1] - m_activation[outputLayer][neuronOut]) * (target[neuronOut-1] - m_activation[outputLayer][neuronOut]);
    m_delta[outputLayer][neuronOut] = (target[neuronOut-1] - m_activation[outputLayer][neuronOut]) * m_activation[outputLayer][neuronOut] * (1.0 - m_activation[outputLayer][neuronOut]);
  }

  
  //error backpropagation to hiddens
  for (unsigned int layer = outputLayer - 1; layer > 0; --layer) {
    for (unsigned int neuronH = 1, neuronHMax = m_activation[layer].size();
	 neuronH < neuronHMax;
	 ++neuronH) {
      m_sumDeltaWeights[layer][neuronH] = 0.0;
      for (unsigned int neuronOut = 0, neuronOutMax = m_activation[layer+1].size();
	   neuronOut < neuronOutMax;
	   ++neuronOut) {
	m_sumDeltaWeights[layer][neuronH] += m_weights[layer][neuronH][neuronOut]*m_delta[layer+1][neuronOut];
      }
      m_delta[layer][neuronH] = m_sumDeltaWeights[layer][neuronH] * m_activation[layer][neuronH] * (1.0 - m_activation[layer][neuronH]);
    }
  }

  //update weights
  for (unsigned int layer = 1, layerMax = m_activation.size(); layer < layerMax; ++layer) {
    for (unsigned int neuronOut = 1/*0?*/, neuronOutMax = m_activation[layer].size();
	 neuronOut < neuronOutMax;
	 ++neuronOut) {
      m_deltaWeights[layer-1][0][neuronOut] = (eta * m_delta[layer][neuronOut]) + (alpha * m_deltaWeights[layer-1][0][neuronOut]);
      m_weights[layer-1][0][neuronOut] += m_deltaWeights[layer-1][0][neuronOut];
      for (unsigned int neuronIn = 0, neuronInMax = m_activation[layer-1].size();
	   neuronIn < neuronInMax;
	   ++neuronIn) {
	m_deltaWeights[layer-1][neuronIn][neuronOut] = (eta * m_activation[layer-1][neuronIn] * m_delta[layer][neuronOut]) + (alpha * m_deltaWeights[layer-1][neuronIn][neuronOut]);
	m_weights[layer-1][neuronIn][neuronOut] += m_deltaWeights[layer-1][neuronIn][neuronOut];
      }
    }
  }
  
  return error;
}

void NeuralNetwork::MLP::save(const std::string& fileName) {
}

void NeuralNetwork::MLP::load(const std::string& fileName) {
}
