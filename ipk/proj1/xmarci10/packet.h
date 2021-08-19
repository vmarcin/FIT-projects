#include <stdio.h>

#define PAYLOAD_SIZE 256 

enum message_type {
	C_HELLO,	//client identifies self ('password' in payload)
	S_ACK,		//acknowledgment of server
	RQ_INFO,	//request for info about 'user' (user name in payload)
	RQ_HOME,	//request for 'user's' home directory (user name in payload)
	RQ_LOGINS,//request for list of logins starts with given prefix or all logins (prefix in payload)
	ERROR,		//server error 
	S_RES,		//server sends results
	D_SEN,		//data are sending
	END				//client finish
};

typedef struct packet {
	message_type type;	
	char payload[PAYLOAD_SIZE];
}message;
