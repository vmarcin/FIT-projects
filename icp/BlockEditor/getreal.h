/**
* @file getreal.h
* @brief GetReal interface
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#ifndef GETREAL_H
#define GETREAL_H

#include "block.h"

class GetReal: public Block
{
public:
    GetReal();
    void Execute();
};

#endif // GETREAL_H
