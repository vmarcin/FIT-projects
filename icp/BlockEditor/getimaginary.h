/**
* @file getimaginary.h
* @brief Getimaginary interface
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#ifndef GETIMAGINARY_H
#define GETIMAGINARY_H

#include "block.h"

class GetImaginary: public Block
{
public:
    GetImaginary();
    void Execute();
};

#endif // GETIMAGINARY_H
