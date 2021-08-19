/**
* @file subractor.h
* @brief Subtractor interface
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#ifndef SUBTRACTOR_H
#define SUBTRACTOR_H

#include "block.h"
#include "port.h"
#include <complex>
#include <vector>
#include "arithmblock.h"

class Subtractor: public ArithmBlock
{
public:
    Subtractor();
    void Execute();
};

#endif // SUBTRACTOR_H
