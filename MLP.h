#ifndef MLP_H_
#define MLP_H_

#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>

namespace NeuralNetwork {
  
  const static double eta = 0.5;
  const static double alpha = 0.5;
  const static double smallwt = 0.5;
  
  enum class NeuronType {
    INPUT,
      HIDDEN,
      OUTPUT
      };
  
  class MLP {
    
  public:
    MLP();
    const std::vector<double>& process(const std::vector<double>& input);
    double train(const std::vector<double>& input, const std::vector<double>& target);
    void add(const NeuronType& nt);
    void save(const std::string& fileName);
    void load(const std::string& fileName);
  private:
    std::vector<double> m_input;
    std::vector< std::vector<double> > m_weightsIH;
    std::vector< std::vector<double> > m_deltaWeightsIH;
    std::vector<double> m_hidden;
    std::vector<double> m_sumH;
    std::vector<double> m_deltaH;
    std::vector< std::vector<double> > m_weightsHO;
    std::vector< std::vector<double> > m_deltaWeightsHO;
    std::vector<double> m_output;
    std::vector<double> m_sumO;
    std::vector<double> m_sumDOW;
    std::vector<double> m_deltaO;
  };

};

#endif /* !MLP_H_ */
