/***********************************
 * Title: Socket class
 * Author: Ryan Lattrel
 * Date: 11/16/2012
 * Avaliability: https://www.egr.msu.edu//classes/ece480/capstone/fall12/group02/documents/Ryan-Lattrel_App-Note.pdf
************************************/

#include "Socket.h"

using namespace std;

Socket::Socket() {
	len = sizeof(sa);
}

//Create a socket given a file descriptor
Socket::Socket(int fd) {
	socket_fd = fd;
	len = sizeof(sa);
}

Socket::~Socket() {
	this->close_socket();
}

//Send a TCP message on socket
int Socket::send_tcp_message(char *msg, size_t msg_size) {
	int send_bytes = send(socket_fd, msg, msg_size, 0);
	return send_bytes;
}

//Receive a TCP message on socket
int Socket::receive_tcp_message(char *buffer, size_t buffer_size) {
	int recv_bytes = recv(socket_fd, buffer, buffer_size, 0);
	return recv_bytes;
}

//Start a TCP server on given port
bool Socket::start_tcp_server(int port) {
	//create socket
	if( (socket_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) 
		return false;

	//setup socket
	sa.sin_family = AF_INET;
	sa.sin_addr.s_addr = htonl(INADDR_ANY);
	sa.sin_port = htons(port);
		
	//bind socket to addr
	if( (bind(socket_fd, (struct sockaddr*)&sa, len)) < 0) 
		return false;
	
	if( (listen(socket_fd, LISTEN_QUEUE_NUMBER)) < 0)
		return false;
	return true;
}

//Waits for a TCP connection and creates a new socket
Socket* Socket::accept_tcp_connection() {
	Socket *new_socket = NULL;
	int accept_connection_fd;
	if( (accept_connection_fd = accept(socket_fd, (struct sockaddr*)0, (socklen_t*)0)) != -1)	
		new_socket = new Socket(accept_connection_fd);
	return new_socket;
}

//Connects to a TCP server on given port and hostname
bool Socket::connect_to_tcp_server(int port, const char *name) {
	hostent *hp;
	//create socket
	if((socket_fd = socket(AF_INET, SOCK_STREAM, 0)) <= 0) 
		return false;

	//setup socket
	sa.sin_family = AF_INET;
	if((hp = gethostbyname(name)) == NULL) 
		return false;
	memcpy(&sa.sin_addr, hp->h_addr, hp->h_length);	
	sa.sin_port = htons(port);	

	//connect to server
	if((connect(socket_fd, (struct sockaddr*)&sa, len)) != 0 ) {
		close(socket_fd);
		return false;
	}

	return true;	
}

//close a socket
int Socket::close_socket() {
	close(socket_fd);
	return 0;
}
