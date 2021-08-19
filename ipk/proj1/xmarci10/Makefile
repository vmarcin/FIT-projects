CPP= g++
LDLIBS = -static-libstdc++
REMOVE = rm -f

.PHONY: all clean

all: ipk-client ipk-server
#linking
ipk-client: client.o Socket.o
	$(CPP) client.o Socket.o $(LDLIBS) -o ipk-client
ipk-server: server.o Socket.o
	$(CPP) server.o Socket.o $(LDLIBS) -o ipk-server
#compilation
client.o: client.cpp Socket.h packet.h
	$(CPP) -c $< -o $@
server.o: server.cpp Socket.h packet.h
	$(CPP) -c $< -o $@
Socket.o: Socket.cpp Socket.h
	$(CPP) -c $< -o $@

clean: 
	$(REMOVE) *.o ipk-client ipk-server
