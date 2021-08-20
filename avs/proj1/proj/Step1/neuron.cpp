/*
 * Architektury výpočetních systémů (AVS 2019)
 * Projekt c. 1 (ANN)
 * Login: xmarci10
 */

#include <cstdlib>
#include "neuron.h"

float evalNeuron(
  size_t inputSize,
  size_t neuronCount, 
  const float* input, 
  const float* weights, 
  float bias, 
  size_t neuronId 
)
{
  float x = 0.0f;

  for(size_t i = 0; i < inputSize; i++ ) 
      x += input[i] * weights[neuronCount*i + neuronId];
  x += bias;

  return (x >= 0) ? x : 0.0f;  
}
