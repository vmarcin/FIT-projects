/**
* @file port.cpp
* @brief Port implementation
* @author Matúš Liščinský xlisci02, Marcin Vladimír xmarci10
*/
#include "port.h"

/**
 * @brief Port constructor
 */
Port::Port()
{
    this->connected_to=NULL;
}

/**
 * @brief Instert complex number to map
 * @param real real part of complex number
 * @param imaginary imaginary part of complex number
 */
void Port::setComplex(double real, double imaginary){
    this->setReal(real);
    this->setImaginary(imaginary);
}

/**
 * @brief Add new item to port type
 * @param new item
 */
void Port::addTypeKey(std::string key){
    this->keys.push_back(key);
}

/**
 * @brief Insert real part of complex number to map
 * @param real part which will be set
 */
void Port::setReal(double real){
    this->m.insert(std::make_pair("real", real));
}

/**
 * @brief Insert imaginary part of complex number to map
 * @param imaginary part which will be set
 */
void Port::setImaginary(double imaginary){
    this->m.insert(std::make_pair("im", imaginary));
}

/**
 * @brief Set polar form of complex number to map
 * @param magnitude magnitude part of polar form
 * @param phase_angle phase part of polar form
 */
void Port::setPolar(double magnitude, double phase_angle){
    this->m.insert(std::make_pair("magnitude", magnitude));
    this->m.insert(std::make_pair("angle", phase_angle));
}

/**
 * @brief Convert map contain to string
 * @return return created string
 */
std::string Port::mapToString(){
    std::string result;
    if(this->m.empty()){
        for(auto &x : this->keys){
            result.append("  ");
            result.append(x);
            result.append(": -\n");
        }
        return result.substr(0,result.size()-1);
    }
    for(auto &x : this->m){
        result.append("  ");
        result.append(x.first);
        result.append(": ");
        // zaokruhlenie
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << x.second;
        std::string val = stream.str();
        result.append(val);
        result.append("\n");
    }
    return result.substr(0,result.size()-1);
}
