///////////////////////////////////////////////////////////////////////////////////
// School:      Brno University of Technology, Faculty of Information Technology //
// Course:      Network Applications and Network Administration                  //
// Project:     DNS Sniffer                                                      //
// Module:      Syslog                                                           //
// Authors:     Vladimír Marcin     (xmarci10)                                   //
///////////////////////////////////////////////////////////////////////////////////

// C header files
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <netdb.h>
#include <arpa/inet.h>

// users header files
#include "syslog.h"
#include "dns-stats.h"


/**
 * @brief Get the my ip address
 * 
 * @param sock_fd   socket descriptor
 * @param addr_family   adress family
 * 
 * @return std::string  returns string containing ip address of
 *                      'this' computer 
 */
std::string get_my_ip_address(int sock_fd, int addr_family) {
    char my_ip[INET6_ADDRSTRLEN];
    struct sockaddr_in my_sock_addr;
    uint32_t length = sizeof(my_sock_addr);

    getsockname(sock_fd, (struct sockaddr *)&my_sock_addr, &length);
    inet_ntop(addr_family, &my_sock_addr.sin_addr, my_ip, INET6_ADDRSTRLEN);
    
    return std::string(my_ip);
}

/**
 * @brief   get the actual time
 * 
 * @return std::string  returns string containing actaul time
 *                      in iso8601 format  
 */
std::string get_iso8601_time() {
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    long ms = current_time.tv_usec / 1000;

    char time_stamp[TIME_STAMP_LEN];

    strftime(time_stamp, TIME_STAMP_LEN - 5, "%FT%T", gmtime(&current_time.tv_sec));

    char ms_str[6] = {0};
    sprintf(ms_str,".%03ldZ",ms);
	strcat(time_stamp, ms_str);
	
    return std::string(time_stamp);
}

/**
 * @brief   creates syslog message header
 * 
 * @param sock_fd   socket descriptor
 * @param addr_family   address family
 * 
 * @return std::string  returns string containing syslog message header  
 */
std::string syslog_hdr(int sock_fd, int addr_family) {
    std::string sys_hdr;

    sys_hdr += "<134>1 ";
    sys_hdr += get_iso8601_time();
    sys_hdr += " ";
    sys_hdr += get_my_ip_address(sock_fd, addr_family);
    sys_hdr = sys_hdr + " dns-export " + std::to_string(getppid()) + " - - ";

    return sys_hdr;
}

void send_handler(int) {
    pid_t pid;

    /// if fork fails parent process send stats
    if((pid = fork()) > 0) {
        alarm(seconds);
        return;
    }
    send_stats_to_syslog();

    if(pid == 0)
        exit(0);
}

void send_stats_to_syslog() {
    struct addrinfo hints;
    struct addrinfo *result, *rp;
    int sfd, s;

    /* Obtain address(es) matching host/port */

    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_UNSPEC;    /* Allow IPv4 or IPv6 */
    hints.ai_socktype = SOCK_DGRAM; /* Datagram socket */
    hints.ai_flags = 0;
    hints.ai_protocol = 0;          /* Any protocol */

    s = getaddrinfo(syslog_server, "514", &hints, &result);
    if (s != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(s));
        return;
    }

    /* getaddrinfo() returns a list of address structures.
    Try each address until we successfully connect(2).
    If socket(2) (or connect(2)) fails, we (close the socket
    and) try the next address. */

    for (rp = result; rp != NULL; rp = rp->ai_next) {
        sfd = socket(rp->ai_family, rp->ai_socktype,
                    rp->ai_protocol);
        if (sfd == -1)
            continue;

        if (connect(sfd, rp->ai_addr, rp->ai_addrlen) != -1)
            break;                  /* Success */

    }

    if (rp == NULL) {               /* No address succeeded */
               fprintf(stderr, "Could not connect\n");
               exit(EXIT_FAILURE);
    }

    /**
     * Create message to send
     */
    std::string message = syslog_hdr(sfd, rp->ai_family);
    std::string stat = {};
    uint32_t message_len = message.length();

    for(auto stats_line = stats.begin(); stats_line != stats.end(); stats_line++) {
        stat += stats_line->first;
        stat += " ";
        stat += std::to_string(stats_line->second);
        stat += "\n";

        if(message_len + stat.length() < 1024) {
            message += stat;
            message_len = message.length();
        }
        else {
            if(send(sfd, message.c_str(), message_len, 0) < 0) {
                fprintf(stderr,"send(): error while sending stats!\n");
                return;
            }
            message = syslog_hdr(sfd, rp->ai_family);
            message += stat;
            message_len = message.length();
        }
        stat = {};
    }

    if(send(sfd, message.c_str(), message_len, 0) < 0) {
                fprintf(stderr,"send(): error while sending stats!\n");
                return;
    }

    freeaddrinfo(result);           /* No longer needed */
    close(sfd);
}
