/**
* @file createcomplex.h
* @brief Create complex interface
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#ifndef CREATECOMPLEX_H
#define CREATECOMPLEX_H

#include "block.h"
#include <string>

class CreateComplex:public Block
{
public:
    CreateComplex();
    void Execute();
};

#endif // CREATECOMPLEX_H
