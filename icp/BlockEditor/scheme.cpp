/**
* @file scheme.cpp
* @brief Scheme implementation
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#include "scheme.h"
#include <cstdio>
#include <fstream>
#include <iostream>

int Scheme::idGenerator = 1;

/**
 * @brief Scheme::addBlock
 * @param b
 */
void Scheme::addBlock(Block * b){
    this->Blocks.push_back(b);
}

/**
 * @brief Scheme::addConnection
 * @param c
 */
void Scheme::addConnection(Connection * c){
    this->Connections.push_back(c);
}

/**
 * @brief Scheme::saveScheme
 * @param filename
 */
void Scheme::saveScheme(std::string filename){
    std::string str;
    std::ofstream out (filename, std::ofstream::out);
    for(Block *b : this->Blocks){
        str.append(std::to_string(b->type).append(" "));
        str.append(std::to_string(b->position[0]).append(" "));
        str.append(std::to_string(b->position[1]));
        str.append("\n");
    }
    str.append("#\n");
    for(Connection* c : this->Connections){
        str.append(std::to_string(this->getIndexOfBlock(c->output->in_block)).append(" "));
        str.append(std::to_string(this->getIndexOfBlock(c->input->in_block)).append(" "));
        if(c->input->in_block->inPorts.size() == 1){
            str.append(std::to_string(15));
        }else if(&c->input->in_block->inPorts[0] == c->input){
            str.append(std::to_string(0));
        }else{
            str.append(std::to_string(30));
        }
        str.append("\n");
    }
    out << str;
    out.close();
}

/**
 * @brief Scheme::getIndexOfBlock
 * @param b
 * @return
 */
int Scheme::getIndexOfBlock(Block * b){
    int index = 0;
    for(Block * t : this->Blocks){
        if(t == b)
            return index;
        index++;
    }
    return -1;
}

/**
 * @brief Scheme::transitiveClosure
 * @param graph
 * @param n
 * @return
 */
int Scheme::transitiveClosure(int * graph, int n){
    int reach[n][n],i, j, k;

    for(int a = 0; a < n; a++){
        for(int b = 0; b < n; b++){
            reach[a][b] = graph[a*n+b];
        }
    } 
    for (k = 0; k < n; k++) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                reach[i][j] = reach[i][j] || (reach[i][k] && reach[k][j]);
                if(i == j && reach[i][j] ==1 )
                    return -1;
            }
        }
    }
        return 0;
}

/**
 * @brief Scheme::checkForCycle
 * @return
 */
int Scheme::checkForCycle(){
    int n = this->Blocks.size();
    int matrix[n*n] = {0};
    for(Connection * c : this->Connections){
        int input_idx = getIndexOfBlock(c->input->in_block);
        int output_idx = getIndexOfBlock(c->output->in_block);
        matrix[output_idx * n + input_idx] = 1;
   }
    int result = transitiveClosure(matrix, n);
    return result;
}

/**
 * @brief Scheme::run
 * @param recursive
 * @return
 */
SchemeItem* Scheme::run(bool recursive){
    // obmedzujuca podmienka
    int cond = 1;
    for(Block *b : this->Blocks){
        for (Port &p : b->outPorts){
            if(p.m.empty())
                cond = 0;
        }
    }
    if(cond)
        return nullptr;
    int flag;
    for(Block * b : this->Blocks){
        flag = 0;
        for (Port &p : b->inPorts){
            b->Load(); // vzdy sa pokusi nacitat input z outputu
            if (p.m.size() == 0){
                flag=1; break;
            }
        }
        bool emptyOutput = b->outPorts[0].m.empty();

        if(flag == 0 && emptyOutput){ // ak vsetky vstupne porty naplnene vykona sa
            b->Execute();
            if(!recursive)
                return b->item;
            //std::cout << b->name << std::endl;
        }
    }
    if(recursive)
        run(true);
    return nullptr;
}

/**
 * @brief Scheme::clearPorts
 */
void Scheme::clearPorts(){
    for(Block *b : this->Blocks){
        for (Port &op : b->outPorts){
            op.m.clear();
        }
        for (Port &ip : b->inPorts){
            ip.m.clear();
        }
    }
}

/**
 * @brief Scheme::removeConnection
 * @param c
 */
void Scheme::removeConnection(Connection *c){
    std::vector<Connection *>::iterator it = find(this->Connections.begin(), this->Connections.end(), c);
    c->input->connected_to=NULL;
    if ( it != this->Connections.end() )
         this->Connections.erase(it);
}

/**
 * @brief Scheme::removeConnectionsOfBlock
 * @param b
 */
void Scheme::removeConnectionsOfBlock(Block *b){
    for(Connection *c : this->Connections){
        if(c->input->in_block == b || c->output->in_block == b)
            removeConnection(c);
    }
}

/**
 * @brief Scheme::removeBlock
 * @param b
 */
void Scheme::removeBlock(Block * b){
    removeConnectionsOfBlock(b);
    std::vector<Block *>::iterator it = find(this->Blocks.begin(), this->Blocks.end(), b);
    if ( it != this->Blocks.end() )
         this->Blocks.erase(it);
}

/**
 * @brief Scheme::clearScheme
 */
void Scheme::clearScheme(){
    std::vector<Block*> help = this->Blocks;
    for(Block *b: help){
        removeBlock(b);
    }
}
