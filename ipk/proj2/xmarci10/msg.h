#ifndef DHCP_MSG
#define DHCP_MSG

/**
 * @brief Function puts together whole packet and sends it 
 * 
 * @param sockfd    socket to send a packet
 * @param ifindex   index of interface to send on
 * @param srcmac    MAC address of interface to send on
 */
void send_discover_msg(int sockfd, int ifindex, const unsigned char* srcmac);



#endif
