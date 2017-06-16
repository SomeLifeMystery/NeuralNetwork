#include <iostream>
#include <cmath>

#include "MLP.h"

NeuralNetwork::MLP::MLP() {
  std::vector<double>tmp;

  m_input.push_back(1.0);//bias
  m_weightsIH.push_back(tmp);
  m_deltaWeightsIH.push_back(tmp);
  m_hidden.push_back(1.0);//bias
  m_sumH.push_back(0.0);
  m_weightsIH[0].push_back((double)((std::rand() % 2400) - 1200) / 1000.0);
  m_weightsHO.push_back(tmp);
  m_sumDOW.push_back(0.0);
  m_deltaH.push_back(0.0);
  m_deltaWeightsIH[0].push_back(0.0);
  m_deltaWeightsHO.push_back(tmp);
}

const std::vector<double>& NeuralNetwork::MLP::process(const std::vector<double>& input) {
  if (input.size() != m_input.size() -1) return input;
  for (int iI = 1, iLI = m_input.size(); iI < iLI; ++iI) m_input[iI] = input[iI-1];

  //hidden activations
  for (int hI = 1, hLI = m_hidden.size(); hI < hLI; ++hI) {
    m_sumH[hI] = m_weightsIH[0][hI];
    for (int iI = 0, iLI = m_input.size(); iI < iLI; ++iI) m_sumH[hI] += m_input[iI]*m_weightsIH[iI][hI];
    m_hidden[hI] = 1.0 / (1.0 + std::exp(-m_sumH[hI]));
  }

  //output activations
  for (int oI = 0, oLI = m_output.size(); oI < oLI; ++oI) {
    m_sumO[oI] = m_weightsHO[0][oI];
    for (int hI = 0, hLI = m_hidden.size(); hI < hLI; ++hI) m_sumO[oI] += m_hidden[hI]*m_weightsHO[hI][oI];
    m_output[oI] = 1.0 / (1.0 + std::exp(-m_sumO[oI]));
  }
  return m_output;
}

double NeuralNetwork::MLP::train(const std::vector<double>& input, const std::vector<double>& target) {
  double error = 0.0;
  if (input.size() != m_input.size() -1 || target.size() != m_output.size()) return 0.0;

  process(input);

  //error calculation
  for (int oI = 0, oLI = m_output.size(); oI < oLI; ++oI) {
    error += (target[oI] - m_output[oI]) * (target[oI] - m_output[oI]);
    m_deltaO[oI] = (target[oI] - m_output[oI]) * m_output[oI] * (1.0 - m_output[oI]);
  }

  //error backpropagation to hiddens
  for (int hI = 0, hLI = m_hidden.size(); hI < hLI; ++hI) {
    m_sumDOW[hI] = 0.0;
    for (int oI = 0, oLI = m_output.size(); oI < oLI; ++oI) m_sumDOW[hI] += m_weightsHO[hI][oI]*m_deltaO[oI];
    m_deltaH[hI] = m_sumDOW[hI] * m_hidden[hI] * (1.0 - m_hidden[hI]);
  }

  //update weightsIH
  for (int hI = 0, hLI = m_hidden.size(); hI < hLI; ++hI) {
    m_deltaWeightsIH[0][hI] = (eta * m_deltaH[hI]) + (alpha * m_deltaWeightsIH[0][hI]);
    m_weightsIH[0][hI] += m_deltaWeightsIH[0][hI];
    for (int iI = 0, iLI = m_input.size(); iI < iLI; ++iI) {
      m_deltaWeightsIH[iI][hI] = (eta * m_input[iI] * m_deltaH[hI]) + (alpha * m_deltaWeightsIH[iI][hI]);
      m_weightsIH[iI][hI] += m_deltaWeightsIH[iI][hI];
    }
  }

  //update weightsHO
  for (int oI = 0, oLI = m_output.size(); oI < oLI; ++oI) {
    m_deltaWeightsHO[0][oI] = (eta * m_deltaO[oI]) + (alpha * m_deltaWeightsHO[0][oI]);
    m_weightsHO[0][oI] += m_deltaWeightsHO[0][oI];
    for (int hI = 0, hLI = m_hidden.size(); hI < hLI; ++hI) {
      m_deltaWeightsHO[hI][oI] = (eta * m_hidden[hI] * m_deltaO[oI]) + (alpha * m_deltaWeightsHO[hI][oI]);
      m_weightsHO[hI][oI] += m_deltaWeightsHO[hI][oI];
    }
  }
    
  return error;
}

void NeuralNetwork::MLP::add(const NeuronType& nt) {
  std::vector<double>tmp;
  
  switch (nt) {
  case NeuronType::INPUT:
    m_input.push_back(0.0);
    m_weightsIH.push_back(tmp);
    for (int iI = m_weightsIH.size()-1, hI = m_hidden.size()-1; hI >= 0; --hI) m_weightsIH[iI].push_back((double)((std::rand() % 2400) - 1200) / 1000.0);
    m_deltaWeightsIH.push_back(tmp);
    for (int iI = m_deltaWeightsIH.size()-1, hI = m_hidden.size()-1; hI >= 0; --hI) m_deltaWeightsIH[iI].push_back(0.0);
    break;
  case NeuronType::HIDDEN:
    m_hidden.push_back(0.0);
    m_sumH.push_back(0.0);
    for (int iI = 0; iI < m_weightsIH.size(); ++iI) m_weightsIH[iI].push_back((double)((std::rand() % 2400) - 1200) / 1000.0);
    for (int iI = 0; iI < m_deltaWeightsIH.size(); ++iI) m_deltaWeightsIH[iI].push_back(0.0);
    m_weightsHO.push_back(tmp);
    for (int hI = m_weightsHO.size()-1, oI = m_output.size()-1; oI >= 0; --oI) m_weightsHO[hI].push_back((double)((std::rand() % 2400) - 1200) / 1000.0);
    m_sumDOW.push_back(0.0);
    m_deltaH.push_back(0.0);
    for (int iI = 0; iI < m_weightsIH.size(); ++iI) m_weightsIH[iI].push_back(0.0);
    m_deltaWeightsHO.push_back(tmp);
    for (int hI = m_deltaWeightsHO.size()-1, oI = m_output.size()-1; oI >= 0; --oI) m_deltaWeightsHO[hI].push_back((double)((std::rand() % 2400) - 1200) / 1000.0);
    break;
  case NeuronType::OUTPUT:
    m_output.push_back(0.0);
    m_sumO.push_back(0.0);
    for (int hI = 0; hI < m_weightsHO.size(); ++hI) m_weightsHO[hI].push_back((double)((std::rand() % 2400) - 1200) / 1000.0);
    m_deltaO.push_back(0.0);
    for (int hI = 0; hI < m_deltaWeightsHO.size(); ++hI) m_deltaWeightsHO[hI].push_back(0.0);
    break;
  default:
    break;
  }
}

void NeuralNetwork::MLP::save(const std::string& fileName) {
}

void NeuralNetwork::MLP::load(const std::string& fileName) {
}
