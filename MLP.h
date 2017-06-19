#ifndef MLP_H_
#define MLP_H_

#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>
#include <exception>

using namespace std;

namespace NeuralNetwork {
  
  const static double eta = 0.5;
  const static double alpha = 0.5;
  const static double smallwt = 0.5;
  
  class MLP {
    
  public:
    MLP(unsigned int nb_layers);
    void addNeuron(unsigned int layer);
    void process(const vector<double>& input, vector<double>& output);
    double train(const vector<double>& input,
		 vector<double>& output,
		 const vector<double>& target);
    void save(const string& fileName);
    void load(const string& fileName);
  private:
    vector< vector<double> > m_activation;//[layer][neuron[layer]]
    vector< vector< vector<double> > > m_weights;//[layer][neuron[layer]][neuron[layer+1]]
    vector< vector< vector<double> > > m_deltaWeights;//[layer][neuron[layer]][neuron[layer+1]]
    vector< vector<double> > m_sum;//[layer][neuron[layer]]
    vector< vector<double> > m_delta;//[layer][neuron[layer]]
    vector< vector<double> > m_sumDeltaWeights;//[layer][neuron[layer]]

    class ConstructionException : public std::exception {
      const char* what() const throw() {
	return "Error: MLP must have at least two layers.";
      }
    };
    class ArgumentException : public std::exception {
      const char* what() const throw() {
	return "Error: MLP process/train inputs or/and outputs size differ from network input and output layers.";
      }
    };
  };

};

#endif /* !MLP_H_ */
