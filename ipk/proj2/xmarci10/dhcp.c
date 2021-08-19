#include "libs.h"
#include "dhcp.h"

int make_dhcp_discover(struct dhcp_packet* dhcp, const unsigned char *srcmac) {
	
	dhcp->op = DHCP_OP_BOOTREQUEST;
	dhcp->htype = DHCP_HTYPE_ETHER;
	dhcp->hlen = DHCP_HLEN_ETHER;
	dhcp->xid = rand();
	dhcp->flags = htons(DHCP_FLAGS_BROADCAST);

	memcpy(&dhcp->chaddr, srcmac, DHCP_HLEN_ETHER);
	
	/* dhcp magic cookie */
	dhcp->options[0] = 0x63;
	dhcp->options[1] = 0x82;
	dhcp->options[2] = 0x53;
	dhcp->options[3] = 0x63;

	dhcp->options[4] = DHCP_OPT_MSGTYPE;
	dhcp->options[5] = sizeof(uint8_t);
	dhcp->options[6] = DHCP_DISCOVER_MSG;
	
	/* end mark */
	dhcp->options[7] = 0xff; 

	return sizeof(struct dhcp_packet);
}
