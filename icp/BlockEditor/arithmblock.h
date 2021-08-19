/**
* @file arithmblock.h
* @brief Arithmblock interface
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#ifndef ARITHMBLOCK_H
#define ARITHMBLOCK_H

#include "block.h"
#include <string>

class ArithmBlock:public Block
{
public:
    ArithmBlock(BlockType type);
};

#endif // ARITHMBLOCK_H
