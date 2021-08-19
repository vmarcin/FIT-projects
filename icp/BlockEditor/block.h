/**
* @file block.h
* @brief Block interface
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#ifndef BLOCK_H
#define BLOCK_H

#include <vector>
#include <string>
#include "port.h"

class Port;
class SchemeItem;

class Block
{
public:
    enum BlockType {add, sub, mul, div, gr, gi, ga, cc, cp};
    BlockType type;
    std::vector<Port> inPorts;
    std::vector<Port> outPorts;
    SchemeItem *item;
    int position [2];
    void Load();
    void addInPort(Port p);
    void addOutPort(Port p);
    void setPosition(int x, int y);
    std::string portsInfo();
    virtual void Execute()=0;
};

#endif // BLOCK_H
