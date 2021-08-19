///////////////////////////////////////////////////////////////////////////////////
// School:      Brno University of Technology, Faculty of Information Technology //
// Course:      Network Applications and Network Administration                  //
// Project:     DNS Sniffer                                                      //
// Module:      Packet dissection                                                //
// Authors:     Vladimír Marcin     (xmarci10)                                   //
///////////////////////////////////////////////////////////////////////////////////

// C header files
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <arpa/inet.h>
#include <net/ethernet.h>
#include <netinet/ip.h>
#include <netinet/ip6.h>
#include <netinet/tcp.h>
#include <string.h>

// user's header files
#include "dissect.h"

#define IP6_HLEN 40

std::vector<tcp_dns_message>dns_messages;

uint32_t handle_IPv6_header(const struct pcap_pkthdr *pkthdr, const unsigned char *packet, uint8_t *trans_proto) {
    struct ip6_hdr *ip6h = (struct ip6_hdr *)packet;
    uint8_t next_header = ip6h->ip6_ctlun.ip6_un1.ip6_un1_nxt;
    uint32_t actual_header_length = IP6_HLEN;
    const unsigned char *ptr = packet;
    uint32_t header_length = IP6_HLEN;
    
    while(next_header != IPPROTO_UDP && next_header != IPPROTO_TCP) {
        ptr += actual_header_length;
        if(check_pkt_ptr(packet, ptr, pkthdr->len))
            return 0;
        switch(next_header) {
            case 0: /* Hop-by-Hop options header */
                next_header = ((struct ip6_hbh *)ptr)->ip6h_nxt;
                actual_header_length = ((struct ip6_hbh *)ptr)->ip6h_len + 1; 
                header_length += actual_header_length;
                break;
            case 60: /* Destination options header */
                next_header = ((struct ip6_dest *)ptr)->ip6d_nxt;
                actual_header_length = ((struct ip6_dest *)ptr)->ip6d_len + 1;
                header_length += actual_header_length;
                break;
            case 43: /* Routing header */
                next_header = ((struct ip6_rthdr *)ptr)->ip6r_nxt;
                actual_header_length = ((struct ip6_rthdr *)ptr)->ip6r_len + 1;
                header_length += actual_header_length;
                break;
            case 44: /* Fragment header */
                next_header = ((struct ip6_frag *)ptr)->ip6f_nxt;
                actual_header_length = sizeof(struct ip6_frag);
                header_length += actual_header_length;
                break;
            default:
                break;
        }
    }
    *trans_proto = next_header;
    return header_length;   
}

unsigned char * handle_tcp_packet(const unsigned char *tcp_hdr, uint32_t tcp_length, uint32_t *pay_len) {
    struct tcphdr *tcph;
    uint32_t payload_length;
    uint32_t seq_number;
    uint32_t next_seq_number;
    unsigned char *dns_message;
    unsigned char *tmp;
    bool found = false;
    unsigned i;

    tcph = (struct tcphdr *)(tcp_hdr);

    /// find tcp payload length and compute next seq number
    payload_length = tcp_length - (tcph->doff * 4);
    if(!payload_length) 
        return NULL;
    
    seq_number = ntohl(tcph->seq);
    next_seq_number = (seq_number + payload_length) % 0x100000000;
    
    /// set next_seq_number and save actual tcp message into vector
    dns_message = (unsigned char *)(tcp_hdr + (tcph->doff * 4));  
    if(check_pkt_ptr(tcp_hdr, dns_message, tcp_length))
        return NULL;

    for(i = 0; i < dns_messages.size(); i++) {
       if(dns_messages[i].next_seq_number == seq_number) {
            found = true;
            dns_messages[i].next_seq_number = next_seq_number;
            dns_messages[i].read_dns_data += payload_length;

            /// alloc space for a part of tcp message
            tmp = (unsigned char *)malloc(payload_length);
            memcpy(tmp, dns_message, payload_length);
            dns_messages[i].ptr_to_tcp.push_back(std::make_pair(payload_length, tmp));
            break;
        }
    }
    /// first part of tcp message
    if(found == false) {
        dns_messages.push_back(tcp_dns_message());
        unsigned long index = dns_messages.size() - 1;
        dns_messages[index].dns_data_length = ntohs(*(uint16_t *)dns_message);
        dns_messages[index].next_seq_number = next_seq_number;
        dns_messages[index].read_dns_data = (payload_length - 2);
        
        /// alloc space for the first part of tcp message
        tmp = (unsigned char *)malloc((payload_length - 2));
        memcpy(tmp, (dns_message+2), (payload_length - 2));
        dns_messages[index].ptr_to_tcp.push_back(std::make_pair((payload_length - 2), tmp));
    }

    /// whole tcp message was read
    if(dns_messages[i].read_dns_data == dns_messages[i].dns_data_length) {
        unsigned char *dns_payload = (unsigned char *)malloc(dns_messages[i].dns_data_length);
        unsigned pointer = 0;
        for(unsigned o = 0; o < dns_messages[i].ptr_to_tcp.size(); o++) {
            memcpy((dns_payload + pointer), dns_messages[i].ptr_to_tcp[o].second, dns_messages[i].ptr_to_tcp[o].first);
            free((void *)dns_messages[i].ptr_to_tcp[o].second);
            
            pointer += dns_messages[i].ptr_to_tcp[o].first;
        }
        *pay_len = dns_messages[i].dns_data_length;
        dns_messages.erase(dns_messages.begin() + i);
        return dns_payload;
    }   

    return NULL;
}

int dissect_packet( const struct pcap_pkthdr *pkthdr, const unsigned char *packet,
                    int link_type, struct packet_payload *payload) {
    uint16_t ether_type;
    uint8_t transport_protocol;
    const unsigned char *pkt = packet;
    uint32_t payload_length;
    uint32_t ipv6_hlen;

    switch(link_type) {
        case DLT_EN10MB: /* Ethernet */
            ether_type = ntohs(((struct ether_header *)pkt)->ether_type);
            pkt += ETH_HLEN;
            break;
        case DLT_LINUX_SLL: /* sockaddr_ll "cooked" packet */
            ether_type = ntohs(*((uint16_t *)(pkt + 14)));
            pkt += 16;
            break;
    }
    
    if(check_pkt_ptr(packet, pkt, pkthdr->len)) 
        return 1;

    switch(ether_type) {
        case ETHERTYPE_IP:
            transport_protocol = ((struct iphdr *)pkt)->protocol;
            pkt += (((struct iphdr *)pkt)->ihl * 4); 
            break;
        case ETHERTYPE_IPV6:
            ipv6_hlen = handle_IPv6_header(pkthdr ,pkt, &transport_protocol);
            if(!ipv6_hlen)
                return 1;
            else
                pkt += ipv6_hlen;
            break;
        default:
            return 1;
    }

    if(check_pkt_ptr(packet, pkt, pkthdr->len))
        return 1;

    switch(transport_protocol) {
        case IPPROTO_UDP:
            pkt += 8;
            payload_length = pkthdr->len - (pkt - packet);
            payload->transport_protocol = IPPROTO_UDP;
            if(check_pkt_ptr(packet, pkt, pkthdr->len))
                return 1;
            break;
        case IPPROTO_TCP:
            pkt = handle_tcp_packet(pkt, (pkthdr->len - (pkt - packet)), &payload_length);
            payload->transport_protocol = IPPROTO_TCP;
            break;
        default:
            return 1;
    }

    if(pkt != NULL) {
        payload->payload = pkt;
        payload->payload_length = payload_length; 
    }
    else return 1;

    return 0;
}