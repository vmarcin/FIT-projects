/**
* @file polartocomplex.h
* @brief PolarToComplex interface
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#ifndef POLARTOCOMPLEX_H
#define POLARTOCOMPLEX_H

#include "block.h"
#include <string>
#include <complex>

class PolarToComplex:public Block
{
public:
    PolarToComplex();
    void Execute();

};

#endif // POLARTOCOMPLEX_H
