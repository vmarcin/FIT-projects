/*
 * Architektury výpočetních systémů (AVS 2019)
 * Projekt c. 1 (ANN)
 * Login: xmarci10
 */

#include <cstdlib>
#include "neuron.h"

float evalNeuron(
  size_t inputSize,
  const float* input, 
  const float* weights, 
  float bias
)
{
  float x = 0.0f;
  size_t i;
  #pragma omp simd reduction(+:x) linear(i, input, weights) simdlen(8) aligned(weights)
  for(i = 0; i < inputSize; i++ ) 
      x += input[i] * weights[i];
  x += bias;

  return (x >= 0) ? x : 0.0f;  
}
