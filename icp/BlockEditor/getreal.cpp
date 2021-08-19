/**
* @file getreal.cpp
* @brief GetReal implementation
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#include "getreal.h"

/**
 * @brief GetReal constructor
 */
GetReal::GetReal()
{
    this->type=BlockType::gr;

    Port p1 = Port();
    p1.in_block=this;
    p1.addTypeKey("real");
    p1.addTypeKey("im");
    this->addInPort(p1);

    Port out = Port();
    out.in_block=this;
    out.addTypeKey("real");
    this->addOutPort(out);
}

/**
 * @brief Extract real part of given complex number
 */
void GetReal::Execute(){
    double real = this->inPorts[0].m.at("real");
    this->outPorts[0].setReal(real);
}
