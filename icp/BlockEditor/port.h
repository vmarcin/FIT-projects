/**
* @file port.h
* @brief Port interface
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#ifndef PORT_H
#define PORT_H
#include <string>
#include <map>
#include <sstream>
#include <iomanip>
#include "block.h"

class Block;

class Port
{
public:
    Port * connected_to;
    Block * in_block ;
    std::vector<std::string> keys;
    std::map<std::string,double, std::greater<std::string>> m;

    Port();
    void setComplex(double real, double imaginary);
    void setReal(double real);
    void setImaginary(double imaginary);
    void setPolar(double magnitude, double phase_angle);
    void addTypeKey(std::string key);
    std::string mapToString();
};

#endif // PORT_H
