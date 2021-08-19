/**
* @file polartocomplex.cpp
* @brief PolarToComplex implementation
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#include "polartocomplex.h"

/**
 * @brief PolarToComplex constructor
 */
PolarToComplex::PolarToComplex()
{
    this->type = BlockType::cp;
    Port p1 = Port();

    p1.in_block=this;
    p1.addTypeKey("magnitude");
    p1.addTypeKey("angle");

    this->addInPort(p1);

    Port out = Port();
    out.in_block=this;
    out.addTypeKey("real");
    out.addTypeKey("im");

    this->addOutPort(out);
}

/**
 * @brief Converts polar for to complex form
 */
void PolarToComplex::Execute(){
    std::complex<double> num;
    double magnitude = this->inPorts[0].m.at("magnitude");
    double angle = this->inPorts[0].m.at("angle");

    num = std::polar(magnitude, angle);
    this->outPorts[0].setComplex(num.real(), num.imag());

}
