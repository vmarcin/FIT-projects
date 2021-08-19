/**
* @file divider.h
* @brief Divider interface
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#ifndef DIVIDER_H
#define DIVIDER_H

#include "block.h"
#include "port.h"
#include <complex>
#include <vector>
#include "arithmblock.h"

class Divider: public ArithmBlock
{
public:
    Divider();
    void Execute();
};

#endif // DIVIDER_H
