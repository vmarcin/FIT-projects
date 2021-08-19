CC= gcc
CFLAGS= #-std=c99 -Wall -Wextra -pedantic
LDLIBS=
EXECUTABLE= ipk-dhcpstarve
REMOVE= rm -f
SRC= $(wildcard *.c)
OBJ= $(SRC:%.c=%.o)

.PHONY: all clean

all: $(EXECUTABLE)

#linking
$(EXECUTABLE): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $(LDLIBS) -o $(EXECUTABLE)

#compilation
dhcp.o: dhcp.c dhcp.h libs.h
	$(CC) $(CFLAGS) -c $< -o $@
main.o: main.c msg.h libs.h 
	$(CC) $(CFLAGS) -c $< -o $@
msg.o: msg.c msg.h libs.h dhcp.h packet_headers.h
	$(CC) $(CFLAGS) -c $< -o $@
packet_headers.o: packet_headers.c packet_headers.h libs.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(REMOVE) *.o ipk-dhcpstarve

