/**
* @file block.cpp
* @brief Block implementation
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#include "block.h"
#include <iostream>
#include <string>

/**
 * @brief Load values to input ports
 */
void Block::Load(){

    for (Port &in : this->inPorts) {
        if(in.connected_to){
            in.m = in.connected_to->m;
        }
    }
}

/**
 * @brief Add input port to block
 * @param p port which will be added
 */
void Block::addInPort(Port p){
    this->inPorts.push_back(p);
}

/**
 * @brief Add output port to block
 * @param p port which will be added
 */
void Block::addOutPort(Port p){
    this->outPorts.push_back(p);
}

/**
 * @brief Set position of block in the scene
 * @param x x coordinate of block position
 * @param y y coordinate of block position
 */
void Block::setPosition(int x, int y){
    this->position[0]=x;
    this->position[1]=y;
}

/**
 * @brief Information about ports types and values
 * @return string which contains information about all ports of block
 */
std::string Block::portsInfo(){
    std::string result;
    int index = 0;
    for(Port &p : this->inPorts){
        ++index;
        result.append("Input");
        result.append(std::to_string(index));
        result.append(":\n");
        result.append(p.mapToString());
        result.append("\n");
    }
    index = 0;
    for(Port &p : this->outPorts){
        ++index;
        result.append("Output");
        result.append(std::to_string(index));
        result.append(":\n");
        result.append(p.mapToString());
        result.append("\n");
    }

    return result.substr(0,result.size()-1);
}
