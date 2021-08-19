/**
* @file subtractor.cpp
* @brief Subtractor implementation
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#include "subtractor.h"

Subtractor::Subtractor():ArithmBlock(BlockType::sub){}

/**
 * @brief Subtract two complex numbers
 */
void Subtractor::Execute(){

    std::vector<std::complex<double>> arr;
    for (Port p : this->inPorts) {
        std::complex<double> c(p.m.at("real"),p.m.at("im"));
        arr.push_back(c);
    }

    std::complex<double> result(arr[0].real(),arr[0].imag());

    for(auto &num: arr){
        if(&num == &arr[0])
            continue;
        result -= num;
    }
    this->outPorts[0].setComplex(result.real(), result.imag());
}
