# project: Implementation and breaking of RSA
# author: Vladimir Marcin
# login: xmarci10
# mail: xmarci10@stud.fit.vutbr.cz
# date: 2.5.2020

CXX = gcc
LDLIBS= -lgmp -lm
EXECUTABLE= kry
REMOVE= rm -f
SRC= $(wildcard *.c)
OBJ= $(SRC:%.c=%.o)

.PHONY: all clean

all: $(EXECUTABLE)

#linking
$(EXECUTABLE): $(OBJ)
	$(CXX) -o $@ $^ $(LDLIBS)

#compilation
main.o: main.c rsa.h
	$(CXX) -c $< -o $@
rsa.o: rsa.c rsa.h primes.h
	$(CXX) -c $< -o $@

clean:
	$(REMOVE) *.o $(EXECUTABLE)
