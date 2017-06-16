#include <iostream>

#include "MLP.h"

int main() {
  NeuralNetwork::MLP p;

  std::srand(std::time(0));

  std::vector< std::pair< std::vector<double>, std::vector<double> > > trainingData_xor/*_and_or*/ = {
    { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } },
    { { 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 }, { 1.0, 0.0, 1.0 } },
    { { 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 }, { 1.0, 0.0, 1.0 } },
    { { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }, { 0.0, 1.0, 1.0 } }
  };/**/

  for (int i = trainingData_xor[0].first.size(); i > 0; --i) {
    p.add(NeuralNetwork::NeuronType::INPUT);
    p.add(NeuralNetwork::NeuronType::HIDDEN);
  }
  for (int i = trainingData_xor[0].second.size(); i > 0; --i) {
    p.add(NeuralNetwork::NeuronType::OUTPUT);
  }
  
  for (int i = 0; i < 1000; ++i) {
    for (int j = 0; j < trainingData_xor.size(); ++j) {
      double error = p.train(trainingData_xor[j].first, trainingData_xor[j].second);
      std::cout << error << std::endl;
    }
  }
  for (const auto& trainingData : trainingData_xor) {
    const std::vector<double>& outputs = p.process(trainingData.first);
    std::cout << "inputs(";
    for (const auto& trainingDataInput : trainingData.first) std::cout << trainingDataInput << ", ";
    std::cout << "\b\b) outputs(";
    for (const auto& output : outputs) std::cout << (output > 0.9 ? 1 : 0) << ", ";
    std::cout << "\b\b) targets(";
    for (const auto& trainingDataTarget : trainingData.second) std::cout << trainingDataTarget << ", ";
    std::cout << "\b\b)" << std::endl;
    }
  return 0;
}
