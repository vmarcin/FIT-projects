#include "libs.h"
#include "packet_headers.h"


int construct_eth_header(struct ether_header *eh,
						const unsigned char *srcmac,
						const unsigned char *dstmac) {

	memcpy(&eh->ether_shost, srcmac, sizeof(eh->ether_shost));
	memcpy(&eh->ether_dhost, dstmac, sizeof(eh->ether_dhost));
	eh->ether_type = htons(ETH_P_IP);

	return sizeof(struct ether_header);
}

unsigned short csum(unsigned short *buf, int nwords) {
    unsigned long sum;
    for(sum=0; nwords>0; nwords--)
        sum += *buf++;
    sum = (sum >> 16) + (sum &0xffff);
    sum += (sum >> 16);
    return (unsigned short)(~sum);
}


int construct_ip_header(struct iphdr *iph,
						uint32_t srchost,
						uint32_t dsthost) {
	iph->ihl = 5;
	iph->version = 4;
	iph->tos = 16; // low delay
	iph->protocol = 17; // UDP
	iph->id = htons(54321);
	iph->ttl = 64;
	iph->saddr = srchost;
	iph->daddr = dsthost;
	iph->check = 0;
	
	return sizeof(struct iphdr);
}


int construct_udp_header(struct udphdr *udph) {
	udph->source = htons(68);
	udph->dest = htons(67);
	udph->check = 0;

	return sizeof(struct udphdr);
}
