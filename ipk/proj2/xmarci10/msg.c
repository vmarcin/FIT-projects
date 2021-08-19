#include "libs.h"
#include "msg.h"
#include "dhcp.h"
#include "packet_headers.h"

const unsigned char brd_mac[] = { 0xff, 0xff, 0xff, 0xff, 0xff, 0xff };

void init_soc_addr(struct sockaddr_ll* soc_addr, int ifindex) {
	memset(soc_addr, 0, sizeof(struct sockaddr_ll));
	soc_addr->sll_ifindex = ifindex;
	soc_addr->sll_hatype = 1;
	soc_addr->sll_family = AF_PACKET;
	soc_addr->sll_pkttype = PACKET_BROADCAST;
	soc_addr->sll_halen = ETH_ALEN;
	soc_addr->sll_protocol = ETH_P_IP;
	memcpy(soc_addr->sll_addr, brd_mac, sizeof(brd_mac));
}

void generate_mac(unsigned char *buff) {
	uint16_t rand_value_prefix = (uint16_t)rand();
	uint32_t rand_value_suffix = rand();
	
	memcpy(buff, (unsigned char*)&rand_value_prefix, sizeof(uint16_t));
	memcpy(buff + sizeof(uint16_t), (unsigned char*)&rand_value_suffix, sizeof(uint32_t));
}


void send_discover_msg(int sockfd, int ifindex, const unsigned char* srcmac) {
	unsigned char mac[DHCP_HLEN_ETHER];
	struct sockaddr_ll socket_addr;
	int sent_bytes;
	int msg_len = 0;
	
	generate_mac(mac);

	unsigned char buffer[1024];

	memset(buffer, 0, sizeof(buffer));

	struct ether_header* eth = (struct ether_header*) buffer;
	msg_len += construct_eth_header(eth, srcmac, brd_mac);

	struct iphdr* iph = (struct iphdr*)(buffer + msg_len);
	msg_len += construct_ip_header(iph, 0, 0xffffffff);

	struct udphdr* udph = (struct udphdr*)(buffer + msg_len);
	msg_len += construct_udp_header(udph);

	struct dhcp_packet* udpdata = (struct dhcp_packet*)(buffer + msg_len);
	msg_len += make_dhcp_discover(udpdata, mac);

	/* length of UDP payload and header */
	udph->len = htons((msg_len - sizeof(struct ether_header) - sizeof(struct iphdr)));

	/* length of IP payload and header */
	iph->tot_len = htons(msg_len - sizeof(struct ether_header));
	
	/* calculate IP checksum on completed header */
	iph->check = csum((unsigned short*)(buffer+sizeof(struct ether_header)), sizeof(struct iphdr)/2);

	init_soc_addr(&socket_addr, ifindex);

	sent_bytes = sendto(sockfd, buffer, msg_len, 0, (struct sockaddr*)&socket_addr, sizeof(struct sockaddr_ll));	
	if(sent_bytes <= 0) {
		fprintf(stderr,"ERROR: Nepodarilo sa odoslat DHCP Discover Message!\n");
		exit(1);
	}
}

