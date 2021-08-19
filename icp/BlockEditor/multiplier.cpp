/**
* @file multiplier.cpp
* @brief Multiplier implementation
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#include "multiplier.h"
#include <iostream>

Multiplier::Multiplier():ArithmBlock(BlockType::mul){}

/**
 * @brief Multiplie two complex numbers
 */
void Multiplier::Execute()
{
    std::vector<std::complex<double>> arr;
    for (Port p : this->inPorts) {
        std::complex<double> c(p.m.at("real"),p.m.at("im"));
        arr.push_back(c);
    }

    std::complex<double> result(1.0,0.0);

    for(auto num: arr){
        result *= num;
    }
    this->outPorts[0].setComplex(result.real(), result.imag());
}
