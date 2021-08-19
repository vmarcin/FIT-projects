#ifndef DHCP_P
#define DHCP_P

#define DHCP_DISCOVER_MSG					1
#define DHCP_OP_BOOTREQUEST					1
#define DHCP_HTYPE_ETHER					1
#define DHCP_HLEN_ETHER						6
#define DHCP_FLAGS_BROADCAST				32768	
#define DHCP_OPT_MSGTYPE					53


#define DHCP_OPT_COUNT 8

/**
 * @brief DHCP Packet RFC 2131
 * 
 */
struct dhcp_packet {
	uint8_t op;
	uint8_t htype;
	uint8_t hlen;
	uint8_t hops;
	uint32_t xid;
	uint16_t secs;
	uint16_t flags;
	uint32_t ciaddr;
	uint32_t yiaddr;
	uint32_t siaddr;
	uint32_t giaddr;
	uint8_t chaddr[16];
	uint8_t sname[64];
	uint8_t file[128];
	uint8_t options[DHCP_OPT_COUNT];
};

/**
 * @brief Function creates DHCP discover message and save it to dhcp structure
 * 
 * @param dhcp 		pointer to structure represents dhcp packet
 * @param srcmac 	MAC address of client
 * @return int 
 */
int make_dhcp_discover(struct dhcp_packet* dhcp, const unsigned char *srcmac);

#endif
