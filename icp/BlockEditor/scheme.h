/**
* @file scheme.h
* @brief Scheme interface
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#ifndef SCHEME_H
#define SCHEME_H

#include <vector>
#include "block.h"
#include "connection.h"
#include <algorithm>

class Scheme
{
public:
    std::vector<Block *> Blocks;
    void generateNewId(){id = idGenerator++;}
    int getId(){return id;}
    void addBlock(Block * b);
    void addConnection(Connection * c);
    void saveScheme(std::string filename);
    void loadScheme();
    int getIndexOfBlock(Block * b);
    int transitiveClosure(int * graph, int n);
    int checkForCycle();
    SchemeItem* run(bool recursive);
    void clearPorts();
    void removeConnection(Connection *c);
    void removeConnectionsOfBlock(Block *b);
    void removeBlock(Block * b);
    void clearScheme();
private:
    std::vector<Connection *> Connections;
    static  int idGenerator;
    int id;
};

#endif // SCHEME_H
