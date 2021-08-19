///////////////////////////////////////////////////////////////////////////////////
// School:      Brno University of Technology, Faculty of Information Technology //
// Course:      Network Applications and Network Administration                  //
// Project:     DNS Sniffer                                                      //
// Module:      Main                                                             //
// Authors:     Vladimír Marcin     (xmarci10)                                   //
///////////////////////////////////////////////////////////////////////////////////

//C header files
#include <stdio.h>
#include <getopt.h>
#include <netinet/in.h>
#include <signal.h>
#include <unistd.h>

//C++ header files
#include <string>
#include <vector>

//users header files
#include "dissect.h"
#include "dns-stats.h"
#include "syslog.h"

#define FAILURE 1
#define SUCCESS 0

static struct packet_payload dns_payload;   /// payload of dns packet
static pcap_t *descr = NULL;                /// pcap session descriptor
int seconds;                                /// time when packets are captured
char *syslog_server;                        /// syslog server name or address

/**
 * @brief   function closes session if open and clears all
 *          memory which was allocated
 */
void error_handler() {
    if(descr)
        pcap_close(descr);
    exit(FAILURE);
}

/**
 * @brief   frees all holden resources before application ends
 * 
 */
void sigint_handler(int) {
    if(dns_payload.transport_protocol == IPPROTO_TCP) {
        if(dns_payload.payload)
            free((void *)dns_payload.payload); 
    }
    for(unsigned i = 0; i < dns_messages.size(); i++) {
        for(unsigned j; j < dns_messages[i].ptr_to_tcp.size(); j++) {
            if(dns_messages[i].ptr_to_tcp[j].second)
                free((void *)dns_messages[i].ptr_to_tcp[j].second);
        }
    }
    if(descr)
        pcap_close(descr);

    exit(SUCCESS);
}

/**
 * @brief   Gets next packet and adds informations to stats 
 * 
 * @param   args    user's defined array of arguments
 * @param   pkthdr  structure which holds information about packet
 * @param   packet  pointer to captured packet
 */
void handle_packet(unsigned char *args, const struct pcap_pkthdr *pkthdr, const unsigned char *packet) {
    int link_type = (int)*args;

    if(!pkthdr || !packet)
        return; 

    if (!dissect_packet(pkthdr, packet, link_type, &dns_payload)) {
        process_dns_payload(&dns_payload);
        /**
         * if transport protocol is TCP, space for packet payload is allocated
         * by user so after work we have to free this space
         */
        if(dns_payload.transport_protocol == IPPROTO_TCP) {
            free((void *)dns_payload.payload); 
        }
    }
}

/**
 * @brief   Prints usage
 * 
 * @param   progname application name 
 */
void usage(const char *progname) {
    printf("\n%s [-r file.pcap] [-i interface] [-s syslog-server] [-t seconds]\n"
           "\t-r file.pcap: Process file.pcap file\n"
           "\t-i interface: Interface to capture on (default interface=any)\n"
           "\t-s syslog-server: Hostname/IPv4/IPv6 address of syslog server\n"
           "\t-t seconds: Time for statistics computation (dafault seconds=60)\n", progname);
    exit(FAILURE);
}

int main(int argc, char **argv) {
    char *pcap_file = NULL; /// file for compute offline statistics
    char *dev = NULL; /// interface to capture on
    char errbuf[PCAP_ERRBUF_SIZE];
    struct bpf_program fp;
    int link_type;
	
    seconds = 60; /// set default time for statistics computation
    syslog_server = NULL;

    if(signal(SIGUSR1, print_handler) == SIG_ERR) {
        fprintf(stderr, "signal(): error!\n");
        error_handler();
    }

    if(signal(SIGALRM, send_handler) == SIG_ERR) {
        fprintf(stderr, "signal(): error!\n");
        error_handler();
    }

    if(signal(SIGINT, sigint_handler) == SIG_ERR) {
        fprintf(stderr, "signal(): error!\n");
        error_handler();
    }

    /**
     * Evaluate args by getopt
     */
    int option;
    uint8_t arg_bitmap = 0;
    char *ptr = NULL;

    while( (option = getopt(argc, argv, "r:i:s:t:h")) != -1) {
        switch(option) {
            case 'r':
                if(arg_bitmap & 8){
                    fprintf(stderr,"%s: duplicit option -- 'r'!\n", argv[0]);
                    usage(argv[0]);
                } 
                arg_bitmap |= 1 << 3;
                pcap_file = optarg;
                break;
            case 'i':
                if(arg_bitmap & 4){
                    fprintf(stderr,"%s: duplicit option -- 'i'!\n", argv[0]);
                    usage(argv[0]);
                } 
                arg_bitmap |= 1 << 2;
                dev = optarg;
                break;
            case 't':
                if(arg_bitmap & 2){
                    fprintf(stderr,"%s: duplicit option -- 't'!\n", argv[0]);
                    usage(argv[0]);
                } 
                arg_bitmap |= 1 << 1;
                seconds = strtol(optarg, &ptr, 10);
                if(*ptr != 0){
                    fprintf(stderr,"%s: -t option argument must be number!\n", argv[0]);
                    usage(argv[0]);
                }
                break;
            case 's':
                if(arg_bitmap & 1){
                    fprintf(stderr,"%s: duplicit option -- 's'!\n", argv[0]);
                    usage(argv[0]);
                } 
                arg_bitmap |= 1;
                syslog_server = optarg;
                break;
            case 'h':
                usage(argv[0]);
                break;
            default:
                usage(argv[0]); 
        }
    }
    if (arg_bitmap > 9 || (syslog_server==NULL && arg_bitmap & 2)) {
        fprintf(stderr, "%s: Invalid arguments combination!\n", argv[0]);
        usage(argv[0]);
    }
    /**
     * offline mode
     */
    if(arg_bitmap & 8) {
        descr = pcap_open_offline(pcap_file, errbuf);
        if(descr == NULL) {
            fprintf(stderr, "pcap_open_offline(): %s\n", errbuf);
            error_handler(); 
        }
    }
    /**
     * online mode
     */
    else {
        fprintf(stderr,"Quit the application with CONTROL-C.\n");
        descr = pcap_open_live(dev, BUFSIZ, 1, 1000, errbuf);
        if(descr == NULL) {
            fprintf(stderr, "pcap_open_live(): %s\n", errbuf);
            error_handler(); 
        }
    }

    /**
     * find the link type of packet
     */
    link_type = pcap_datalink(descr);
    if(link_type != DLT_EN10MB && link_type != DLT_LINUX_SLL) {
        fprintf(stderr, "Unsuported link type %d\n", link_type);
        error_handler(); 
    }

    /**
     * set pcap filter
     */
    if(pcap_compile(descr, &fp, "src port 53", 0, PCAP_NETMASK_UNKNOWN) == -1) {
        pcap_perror(descr, (char *)"pcap_compile()");
            error_handler(); 
    }
    if(pcap_setfilter(descr, &fp) == -1) {
        pcap_perror(descr, (char *)"pcap_filter()");
        error_handler(); 
    }

    /**
     * if syslog server is given and online mode is turned on,
     * timeout for packets capturing starts
     */
    if(syslog_server && !(arg_bitmap & 8))
        alarm(seconds);

    /**
     * Grabs packets
     */
    pcap_loop(descr, -1, handle_packet, (unsigned char *)&link_type);

    /** 
     * just offline mode reachs here      
     */
    if(syslog_server == NULL)
        print_stats_map();
    else
        send_stats_to_syslog();

    if(descr)
        pcap_close(descr);

    return SUCCESS;
}
