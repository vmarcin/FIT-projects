#include "Socket.h"
#include <string>
#include "packet.h"

#define SUCC	0
#define ERR		1

//print usage if wrong option is used
void usage(){
	fprintf(stderr,"Usage: ./ipk-client [OPTIONS]\n");
	fprintf(stderr,"\t-h host\tIP address or fully-qualified DNS name\n");
	fprintf(stderr,"\t-p port\tdestination port number <49152, 65535>\n");
	fprintf(stderr,"\t[-n|-f|-l] login\n");
}

//check arguments validity
bool check_arguments(int argc, char *argv[], const char **host, int *port, char **login, int *req) {
	int opt, rep_h = 0, rep_p = 0, rep_o = 0;
	long p;
	char *ptr = NULL;
	if(argc < 6 || argc > 7) return false;
	while( (opt = getopt(argc, argv, "h:p:n:f:l")) != -1 ) {
		switch(opt) {
			case 'h':
				if(++rep_h > 1) return false;
				*host = optarg;
				break;
			case 'p':	
				if(++rep_p > 1) return false;
				p = strtol(optarg, &ptr, 10);
				//check if port number belongs to ephemeral ports (ports for client's programs) 
        		if(*ptr != 0 || p < 49152 || p > 65535) return false;  
        		*port = (int)p; 
        		break; 
			case 'n':
				if(++rep_o > 1) return false;
				*login = optarg;
				*req = 2;
				break;
			case 'f':
				if(++rep_o > 1) return false;
				*login = optarg;
				*req = 3;
				break;
			case 'l':
				if(++rep_o > 1) return false;
				(argc == 6) ? *login = NULL : *login = argv[optind];
				*req = 4;
				break;
			default:
				return false;
		}
	}
	return true;
}

int main(int argc, char *argv[]) {
	const char *server_name_str;
	int tcp_server_port, req, type, res = 0;
	char *login = NULL;
	
	message incoming_message;
	message outgoing_message;
	
	memset(incoming_message.payload, '\0', PAYLOAD_SIZE);
	memset(outgoing_message.payload, '\0', PAYLOAD_SIZE);

	if(!(check_arguments(argc,argv, &server_name_str, &tcp_server_port, &login, &req))) {
		usage();
		return ERR;
	}
	
	Socket* TCPClientSocket = new Socket();

	if(!(TCPClientSocket->connect_to_tcp_server(tcp_server_port, server_name_str))) {
		fprintf(stderr,"ERROR: Unsuccessful attempt to connect the server!\n");
		delete TCPClientSocket;
		return ERR;
	}
	
	//client identificates itself by sending the hello packet 
	outgoing_message.type = C_HELLO;
	strncpy(outgoing_message.payload, "tajneheslo", PAYLOAD_SIZE-1);

	TCPClientSocket->send_tcp_message((char*)&outgoing_message, sizeof(outgoing_message));
	TCPClientSocket->receive_tcp_message((char*)&incoming_message, sizeof(incoming_message));
	
	if(incoming_message.type != S_ACK) {
		fprintf(stderr,"ERROR: Permission denied by server!\n");
		TCPClientSocket->close_socket();
		delete TCPClientSocket;
		return ERR;
	}
	//client sends request to server 
	if(login != NULL) {
		outgoing_message.type = D_SEN;
		int length = strlen(login);
		int bytes_sent, i = 0;
		int copy_data_len = PAYLOAD_SIZE - 1;
		
		//if login is longer than sizeof message it must be sent in more messages
		while(length > 0) {
			memset(outgoing_message.payload, '\0', PAYLOAD_SIZE);
			if(strlen(login+(i*(PAYLOAD_SIZE-1))) <= copy_data_len) 
				outgoing_message.type = (message_type)req;
			strncpy(outgoing_message.payload, login + (i++*(PAYLOAD_SIZE-1)), copy_data_len);
			bytes_sent = TCPClientSocket->send_tcp_message((char*)&outgoing_message, sizeof(outgoing_message));
			length -= copy_data_len;
			if(bytes_sent == 0) break;
			else if(bytes_sent < 0) {
				fprintf(stderr, "ERROR: Error while data was sending!\n");
				TCPClientSocket->close_socket();
				delete TCPClientSocket;
				return ERR;
			}
		}
	}else{
		//if option -l doesn't have value we just send an empty string
		outgoing_message.type = (message_type)req;
		memset(outgoing_message.payload, '\0', PAYLOAD_SIZE);
		strncpy(outgoing_message.payload, "", PAYLOAD_SIZE - 1);
		TCPClientSocket->send_tcp_message((char*)&outgoing_message, sizeof(outgoing_message));
	}

	//message receiving
	do {
		memset(incoming_message.payload, '\0', PAYLOAD_SIZE);
		res = TCPClientSocket->receive_tcp_message((char*)&incoming_message, sizeof(incoming_message));
		if(res == 0) break;
		else if(res < 0) {
			fprintf(stderr, "ERROR: Error while data was receiving!\n");
			TCPClientSocket->close_socket();
			delete TCPClientSocket;
			return ERR;
		}
		type = incoming_message.type;
		if(type == ERROR) {
			fprintf(stderr,"ERROR: Server error (login wasn't find)!\n");
			TCPClientSocket->close_socket();
			delete TCPClientSocket;
			return ERR;
		}
		printf("%s", incoming_message.payload);	
	}while(type == D_SEN);	
	printf("\n");	

	//client sends final ack	
	outgoing_message.type = END;
	memset(outgoing_message.payload, '\0', PAYLOAD_SIZE);
	TCPClientSocket->send_tcp_message((char*)&outgoing_message, sizeof(outgoing_message));

	//client finished
	TCPClientSocket->close_socket();
	delete TCPClientSocket;
	return SUCC;
}
