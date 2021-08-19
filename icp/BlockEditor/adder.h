/**
* @file adder.h
* @brief Adder interface
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#ifndef ADDER_H
#define ADDER_H

#include "block.h"
#include "port.h"
#include <complex>
#include <vector>
#include "arithmblock.h"

class Adder: public ArithmBlock
{
public:
    Adder();
    void Execute();
};

#endif // ADDER_H
