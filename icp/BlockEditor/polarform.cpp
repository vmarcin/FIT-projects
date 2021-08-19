/**
* @file polarform.cpp
* @brief PolarForm implementation
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#include "polarform.h"

/**
 * @brief PolarForm constructor
 */
PolarForm::PolarForm()
{
    this->type=BlockType::ga;
    Port p1 = Port();
    p1.in_block=this;
    p1.addTypeKey("real");
    p1.addTypeKey("im");
    this->addInPort(p1);

    Port out = Port();
    out.addTypeKey("magnitude");
    out.addTypeKey("angle");
    out.in_block=this;
    this->addOutPort(out);
}

/**
 * @brief Create polar form of complex number
 */
void PolarForm::Execute(){
    std::complex<double> num (this->inPorts[0].m.at("real"),this->inPorts[0].m.at("im"));
    this->outPorts[0].setPolar(std::abs(num), std::arg(num));

}
