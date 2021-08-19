/**
* @file createcomplex.cpp
* @brief CreateComplex implementation
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#include "createcomplex.h"

/**
 * @brief CreateComplex constructor
 */
CreateComplex::CreateComplex()
{
    this->type = BlockType::cc;
    Port p1 = Port();
    Port p2 = Port();

    p1.in_block=this;
    p1.addTypeKey("real");

    p2.in_block=this;
    p2.addTypeKey("im");

    this->addInPort(p1);
    this->addInPort(p2);

    Port out = Port();
    out.in_block=this;
    out.addTypeKey("real");
    out.addTypeKey("im");

    this->addOutPort(out);
}

/**
 * @brief Create complex number from given real and imaginary part
 */
void CreateComplex::Execute(){
    double real = this->inPorts[0].m.at("real");
    double imaginary = this->inPorts[1].m.at("im");
    this->outPorts[0].setComplex(real, imaginary);
}
