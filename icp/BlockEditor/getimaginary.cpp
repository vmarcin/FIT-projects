/**
* @file getimaginary.cpp
* @brief GetImaginary implementation
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#include "getimaginary.h"

/**
 * @brief GetImaginary constructor
 */
GetImaginary::GetImaginary()
{
    this->type=BlockType::gi;

    Port p1 = Port();
    p1.in_block=this;
    p1.addTypeKey("real");
    p1.addTypeKey("im");
    this->addInPort(p1);

    Port out = Port();
    out.in_block=this;
    out.addTypeKey("im");
    this->addOutPort(out);
}

/**
 * @brief Extract imaginary part of given complex number
 */
void GetImaginary::Execute(){
    double imaginary = this->inPorts[0].m.at("im");
    this->outPorts[0].setImaginary(imaginary);
}
