/**
* @file arithmblock.cpp
* @brief ArithmBlock implementation
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#include "arithmblock.h"

/**
 * @brief Arithmetic Block constructor
 * @param type of arithmetic block
 */
ArithmBlock::ArithmBlock(BlockType type)
{
    this->type = type;

    Port p1 = Port();
    Port p2 = Port();
    p1.in_block=this;
    p2.in_block=this;
    p1.addTypeKey("real");
    p1.addTypeKey("im");
    p2.addTypeKey("real");
    p2.addTypeKey("im");
    this->addInPort(p1);
    this->addInPort(p2);

    Port out = Port();
    out.in_block=this;
    out.addTypeKey("real");
    out.addTypeKey("im");
    this->addOutPort(out);
}
