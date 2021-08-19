/**
* @file connection.cpp
* @brief Connection implementation
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#include "connection.h"

/**
 * @brief Connection constructor
 * @param output source port of connection
 * @param input destination port of connection
 */
Connection::Connection(Port * output, Port * input)
{
    this->output=output;
    this->input=input;

    if(this->input == NULL || this->output == NULL){
        fprintf(stderr,"NULL dereference.");
        exit(1);
    }

    this->input->connected_to=this->output;
}

/**
 * @brief Check compatiblity of connection (compare types of ports in connection)
 * @return if types are identical return 0 else return -1
 */
int Connection::checkCompatibility()
{
    std::vector<std::string> keys1 = this->output->keys;
    std::vector<std::string> keys2 = this->input->keys;

    if(keys1.size() == keys2.size())
        if(std::equal(keys1.begin(), keys1.end(), keys2.begin()))
            return 0;
    return -1;
}

