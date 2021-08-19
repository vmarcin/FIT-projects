///////////////////////////////////////////////////////////////////////////////////
// School:      Brno University of Technology, Faculty of Information Technology //
// Course:      Network Applications and Network Administration                  //
// Project:     DNS Sniffer                                                      //
// Module:      Stats computation                                                //
// Authors:     Vladimír Marcin     (xmarci10)                                   //
///////////////////////////////////////////////////////////////////////////////////

#ifndef DNS_STATS_H
#define DNS_STATS_H

// C++ header files
#include <map>

// user's header files
#include "dissect.h"

extern std::map<std::string, int>stats; /// DNS stats

/**
 * @brief   after program receive signal 'SIGUSR1' prints stats to stdout
 * 
 */
void print_handler(int);

/**
 * @brief   prints stats to stdout
 * 
 */
void print_stats_map();

/**
 * @brief   process payload of dns message and adds it to stats
 * 
 * @param dns_payload   payload of analyzed packet
 */
void process_dns_payload(struct packet_payload *dns_payload);

#endif /* DNS_STATS_H*/