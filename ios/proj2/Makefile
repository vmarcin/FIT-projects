# Makefile
# Riesenie IOS-proj2, 27.4.2017
# Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
# Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
C=gcc
CFLAGS=-std=gnu99 -Wall -Wextra -pedantic

all: proj2 

proj2.o: proj2.c proj2.h
	$(CC) $(CFLAGS) -c $< -o $@

proj2: proj2.o
	$(CC) $(CFLAGS) $< -o $@ -lpthread -lrt 

clean:
	rm -f *.o proj2 proj2.out
