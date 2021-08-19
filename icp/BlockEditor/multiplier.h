/**
* @file multiplier.h
* @brief Multiplier interface
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#ifndef MULTIPLIER_H
#define MULTIPLIER_H

#include "block.h"
#include "port.h"
#include <complex>
#include <vector>
#include "arithmblock.h"

class Multiplier: public ArithmBlock
{
public:
    Multiplier();
    void Execute();
};

#endif // MULTIPLIER_H
