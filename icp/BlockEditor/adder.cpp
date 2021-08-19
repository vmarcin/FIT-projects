/**
* @file adder.cpp
* @brief Adder implementation
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#include "adder.h"
#include <iostream>

Adder::Adder():ArithmBlock(BlockType::add){}

/**
 * @brief Add two complex numbers
 */
void Adder::Execute(){
    std::vector<std::complex<double>> arr;
    for (Port p : this->inPorts) {
        std::complex<double> c(p.m.at("real"),p.m.at("im"));
        arr.push_back(c);
    }

    std::complex<double> result(0.0,0.0);

    for(auto num: arr){
        result += num;
    }

    this->outPorts[0].setComplex(result.real(), result.imag());
}
