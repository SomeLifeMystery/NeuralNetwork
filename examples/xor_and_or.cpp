#include <iostream>

#include "MLP.h"

int main() {
  std::srand(std::time(0));

  try {
    NeuralNetwork::MLP p(5);

    std::vector< std::pair< std::vector<double>, std::vector<double> > > trainingData_xor = {
      { { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } },
      { { 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 }, { 1.0, 0.0, 1.0 } },
      { { 1.0, 0.0, 1.0, 0.0, 1.0, 0.0 }, { 1.0, 0.0, 1.0 } },
      { { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 }, { 0.0, 1.0, 1.0 } }
    };
    std::vector<double> outputs = trainingData_xor[0].second;

    for (int i = trainingData_xor[0].first.size(); i > 0; --i) {
      p.addNeuron(0);
      p.addNeuron(1);
      p.addNeuron(2);
      p.addNeuron(3);
    }
    for (int i = trainingData_xor[0].second.size(); i > 0; --i) {
      p.addNeuron(4);
    }
  
    for (int i = 0; i < 1000; ++i) {
      for (int j = 0; j < trainingData_xor.size(); ++j) {
	double error = p.train(trainingData_xor[j].first, outputs, trainingData_xor[j].second);
	std::cout << error << std::endl;
      }
    }
    for (const auto& trainingData : trainingData_xor) {
      p.process(trainingData.first, outputs);
      std::cout << "inputs(";
      for (const auto& trainingDataInput : trainingData.first) std::cout << trainingDataInput << ", ";
      std::cout << "\b\b) outputs(";
      for (const auto& output : outputs) std::cout << (output > 0.9 ? 1 : 0) << ", ";
      std::cout << "\b\b) targets(";
      for (const auto& trainingDataTarget : trainingData.second) std::cout << trainingDataTarget << ", ";
      std::cout << "\b\b)" << std::endl;
    }
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
  }
  return 0;
}
