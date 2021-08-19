/**
* @file polarform.h
* @brief Polarform interface
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#ifndef POLARFORM_H
#define POLARFORM_H


#define _USE_MATH_DEFINES
#include <cmath>
#define radiansToDegrees(angleRadians) (angleRadians * 180.0 / M_PI)
#include <complex>

#include "block.h"

class PolarForm: public Block
{
public:
    PolarForm();
    void Execute();
};

#endif // POLARFORM_H
