#ifndef PACKET_HEADERS
#define PACKET_HEADERS

#include "libs.h"

/**
 * @brief Function constructs ethernet header 
 * 
 * @param eh 		pointer to struct which represents ethernet header
 * @param srcmac 	source MAC address
 * @param dstmac 	destination MAC address
 * @return int 		returns size of ethernet header
 */
int construct_eth_header(struct ether_header *eh,
                         const unsigned char *srcmac,
                         const unsigned char *dstmac);
/**
 * @brief Function computes checksum of ip header
 * 
 * @Author Marton Austin
 * @Title Sending raw Ethernet packets from a specific interface in C on Linux
 * @Availability https://austinmarton.wordpress.com/2011/09/14/sending-raw-ethernet-packets-from-a-specific-interface-in-c-on-linux/
 * @Date 14.9.2011
 *
 * @param buf   pointer to ip header 
 * @param nwords    number of words in ip header
 * @return unsigned short	checksum 
 */
unsigned short csum(unsigned short *buf, int nwords);

/**
 * @brief Function constructs ip header
 * 
 * @param iph 		pointer to ip header struct
 * @param srchost 	source IP address
 * @param dsthost 	destination IP address
 * @return int 		return size of ip header
 */
int construct_ip_header(struct iphdr *iph,
                        uint32_t srchost,
                        uint32_t dsthost);
/**
 * @brief  Function constructs udp header
 *  
 * @param udph      pointer to udp header struct 
 * @return int      return size of udp header
 */
int construct_udp_header(struct udphdr *udph);

#endif
