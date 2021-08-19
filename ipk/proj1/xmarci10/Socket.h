#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstring>
#include <netdb.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define LISTEN_QUEUE_NUMBER 256 

class Socket {
	public:
		Socket();
		Socket(int fd);
		~Socket();	//destructor
		int send_tcp_message(char *msg, size_t msg_size);
		bool start_tcp_server(int port);
		Socket* accept_tcp_connection();
		bool connect_to_tcp_server(int port, const char *name);
		int receive_tcp_message(char *buffer, size_t buffer_size);
		int close_socket();

	private:
		int socket_fd;
		sockaddr_in sa;
		socklen_t len;
};

