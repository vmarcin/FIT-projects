///////////////////////////////////////////////////////////////////////////////////
// School:      Brno University of Technology, Faculty of Information Technology //
// Course:      Network Applications and Network Administration                  //
// Project:     DNS Sniffer                                                      //
// Module:      Syslog                                                           //
// Authors:     Vladimír Marcin     (xmarci10)                                   //
///////////////////////////////////////////////////////////////////////////////////

#ifndef SYSLOG_H
#define SYSLOG_H

#define TIME_STAMP_LEN 25

extern int seconds; /// time when packets are captured
extern char *syslog_server; /// syslog server name or address

/**
 * @brief   after timeout ('seconds') sends stats to 'syslog server'
 * 
 */
void send_handler(int);

/**
 * @brief   sends stats to 'syslog server'
 * 
 */
void send_stats_to_syslog();

#endif //SYSLOG_H