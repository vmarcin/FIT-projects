/**
* @file connection.h
* @brief Connection interface
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#ifndef CONNECTION_H
#define CONNECTION_H

#include "block.h"
#include "port.h"


class Connection
{
public:
    Connection(Port * output, Port * input);
    int checkCompatibility();

    Port * output;
    Port * input;
};

#endif // CONNECTION_H
