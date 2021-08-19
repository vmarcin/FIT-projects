# Server-Client application 
A simple client/server application in C/C ++ that provides information about server-side users to the client.
### Run server
```sh
$ ./ipk-server -p port
```
- port : [empheral port number](http://www.steves-internet-guide.com/tcpip-ports-sockets/) <49152-65535> 

### Run client
```sh
$ ./ipk-client -h host -p port [-n|-f|-l] login
```
  - host : (IP address or fully-qualified DNS name) server identification 
  - port : destination port number
  - -n : user full name will be returned
  - -f : information about home directory will be returned
  - -l : list of all users will be returned (login is optional) if login is given it will be used as prefix for users pick
  - login : determines the user login name for the above operations

### Limitations
 - the application only supports IPv4 protocol

### License
*Free Software, Hell Yeah!*
