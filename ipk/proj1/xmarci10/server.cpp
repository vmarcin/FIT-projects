#include "Socket.h"
#include "packet.h"
#include <pwd.h>
#include <string>
#include <fstream>

#define SUCC	0
#define ERR		1

//print usage if wrong option is used
void usage() {
	fprintf(stderr, "Usage: ./ipk-server [OPTIONS]\n");
	fprintf(stderr, "\t-p port\tnumber from interval <49152, 65535>\n");
}

//check argumets validity
bool check_arguments(int argc, char *argv[], int *port) {
	int opt, rep = 0;
	long p;
	char *ptr = NULL;
	if(argc != 3) return false;
	while( (opt = getopt(argc,argv, "p:")) != -1 ) {
		switch(opt) {
			case 'p':
				if(++rep > 1) return false;
				p = strtol(optarg, &ptr, 10);	
				if(*ptr != 0 || p < 49152 || p > 65535) return false;
				*port = (int)p;
				break;
			default:
				return false;
		}
	}
	return true;
}

//return list of user's logins in /etc/passwd
//if login is given it return just users with given 'login' or prefix of given login
const char *get_users_logins(std::string login) {
	std::string line, first_token, mess;
	std::ifstream fp("/etc/passwd");	
	
	if(!(fp.is_open())) return NULL;
	
	while(std::getline(fp,line)) {
		if(line.find(login) == 0 ) {
			first_token = line.substr(0, line.find(':'));
			mess.append(first_token);
			mess.append("\n");
		}
	}
	//strip last character from string ('\n')
	mess = mess.substr(0, mess.size()-1);
	fp.close();
	if(mess.empty())
		return NULL;
	return strdup(mess.c_str());	
}

//return home directory of user with login 'login'
char *get_dir(const char *login) {
	struct passwd *p;
	if(!(p = getpwnam(login))) return NULL;
	else
		return p->pw_dir;
}

//return full name of user with login 'login'
char *get_gecos(const char *login) {
	struct passwd *p;
	if(!(p = getpwnam(login))) return NULL;
	else
		return p->pw_gecos;
}

int main(int argc, char *argv[]) {
	int port=0, length, bytes_sent, i=0;
	int copy_data_len = PAYLOAD_SIZE - 1;
	int res = 0;
	std::string str = "";
	message_type type;
	const char *msg;
	pid_t pid;
	bool end = 0;
	
	message incoming_message;
	message outgoing_message;
	
	if(!(check_arguments(argc,argv, &port))) {
		usage();
		return ERR;
	}	

	Socket *TCPServer = new Socket();
	if(!(TCPServer->start_tcp_server(port))) {
		fprintf(stderr,"ERROR: Unsuccessful attempt to create server!\n");
		delete TCPServer;
		return ERR;
	}
	
	while(1) {
		Socket* client_connection = TCPServer->accept_tcp_connection();
		if(client_connection == NULL) {
			fprintf(stderr, "ERROR: Server didn't accept connection!\n");	
			return ERR;
		}
		pid = fork();			
		if(pid == 0) { //child process
		
			for(;;) {
				//message receiving
				do {
					memset(incoming_message.payload, '\0', PAYLOAD_SIZE);	
					res = client_connection->receive_tcp_message((char*)&incoming_message, sizeof(incoming_message));	
					if(res == 0) break;	
					if(res < 0) {
						fprintf(stderr, "ERROR: Error while data was receiving!\n");
						client_connection->close_socket();
						delete client_connection;
						return ERR;	
					}
					type = incoming_message.type;
					str.append(incoming_message.payload);
				}while(type == D_SEN);	
				if(res == 0) break;

				//handle different message types
				switch(type) {
					case C_HELLO:
						if(!strcmp(str.c_str(),"tajneheslo")) 
							outgoing_message.type = S_ACK;
						else 
							outgoing_message.type = ERROR;
						msg = NULL;
						break;
					case RQ_INFO:
						msg = get_gecos(str.c_str());
						if(msg == NULL) { 
							outgoing_message.type = ERROR;
							break;
						}
						if(!strcmp(msg,"")) {
							msg = NULL;
							outgoing_message.type = S_RES;
							break;
						}
						outgoing_message.type = D_SEN; 
						break;
					case RQ_HOME:
						msg = get_dir(str.c_str());
						if(msg == NULL){ 
							outgoing_message.type = ERROR;
							break;
						}
						if(!strcmp(msg,"")) {
							msg = NULL;
							outgoing_message.type = S_RES;
							break;
						}
						outgoing_message.type = D_SEN;
						break;
					case RQ_LOGINS:
						msg = get_users_logins(str);
						if(msg == NULL) 
							outgoing_message.type = ERROR;
						else	
							outgoing_message.type = D_SEN;
						break;
					case END:
						end = 1;
						break;
				}
				
				if(end) break;
				str.clear();
				//sending message
				if(msg != NULL) {	
					length = strlen(msg);
					while(length > 0) {
						memset(outgoing_message.payload, '\0', PAYLOAD_SIZE);
						if(strlen(msg+(i*(PAYLOAD_SIZE-1))) <= copy_data_len)
							outgoing_message.type = (message_type)S_RES;
						strncpy(outgoing_message.payload, msg + (i++*(PAYLOAD_SIZE-1)), copy_data_len);	
						bytes_sent = client_connection->send_tcp_message((char*)&outgoing_message, sizeof(outgoing_message));
						length -= copy_data_len;
						if(bytes_sent == 0) break;
						else if(bytes_sent < 0) {
							fprintf(stderr, "ERROR: Error while data was sending!\n");
							client_connection->close_socket();
							delete client_connection;
							return ERR;
						}
					}
					i = 0;
				}
				else {
					memset(outgoing_message.payload, '\0', PAYLOAD_SIZE);
					client_connection->send_tcp_message((char*)&outgoing_message, sizeof(outgoing_message));
				}
				if(outgoing_message.type == ERROR)
					break;
			}
			client_connection->close_socket();
			delete client_connection;
			exit(0); //child process end		
		}else if(pid == -1) {
			fprintf(stderr, "ERROR: Process create error!\n");
			client_connection->close_socket();
			delete TCPServer;
			delete client_connection;
			return ERR;
		}else {
			client_connection->close_socket();
			delete client_connection;
		}
		//sleep ensures that server does not eat up all of CPU processing
		sleep(1);
	}
}
