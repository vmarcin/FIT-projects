///////////////////////////////////////////////////////////////////////////////////
// School:      Brno University of Technology, Faculty of Information Technology //
// Course:      Network Applications and Network Administration                  //
// Project:     DNS Sniffer                                                      //
// Module:      Packet dissection                                                //
// Authors:     Vladimír Marcin     (xmarci10)                                   //
///////////////////////////////////////////////////////////////////////////////////

#ifndef DISSECT_H
#define DISSECT_H

// C++
#include <pcap/pcap.h>
#include <stdint.h>
#include <stdbool.h>
#include <vector>

/**
 * @brief holds dissected packet data
 */
struct packet_payload {
    uint8_t transport_protocol;
    uint32_t payload_length;    /// DNS payload lenght
    const unsigned char *payload;   /// pointer to DNS header
};

/**
 * @brief holds segmented tcp message
 */
struct tcp_dns_message{
    uint16_t read_dns_data; /// length of data which i already read 
    uint16_t dns_data_length; /// length of dns data including dns header length 
    uint32_t next_seq_number; /// seq number which i'm waiting for 
    std::vector<std::pair <unsigned long, unsigned char *>>ptr_to_tcp; /// vector of pointer to segments of tcp message
};

extern std::vector<tcp_dns_message>dns_messages; /// vector of all DNS messages received via TCP

/**
 * @brief   checks validity of 'pkt_curr' pointer
 * 
 * @param pkt_start pointer to packet's start
 * @param pkt_curr  pointer to acutal position in packet
 * @param pkt_len   length of packet
 * 
 * @return true     if returns true the 'pkt_curr' is beyond the bounds of packet
 * @return false    if returns false everything alright
 */
inline bool check_pkt_ptr(  const unsigned char *pkt_start, 
                            const unsigned char *pkt_curr,
                            uint32_t pkt_len) {
    if(pkt_curr >= (pkt_start + pkt_len)) 
        return true;
    else 
        return false;
}

/**
 * @brief   dissects packet's headers
 * 
 * @param pkthdr    information about packet given by libpcap
 * @param packet    pointer to packet's start
 * @param link_type type of link layer
 * @param payload   dissected packet struct to fill in
 * 
 * @return int      returns 0 in case of succes and 1 otherwise
 */
int dissect_packet( const struct pcap_pkthdr *pkthdr, 
                    const unsigned char *packet,
                    int link_type,
                    struct packet_payload *payload);

#endif /*DISSECT_H*/