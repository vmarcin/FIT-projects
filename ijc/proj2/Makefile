# Makefile
# Riesenie IJC-DU2, 22.4.2017
# Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
# Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 
CC      = gcc
CPP     = g++
CFLAGS  = -std=c99 -g -Wall -Wextra -pedantic
CPPFLAGS= -std=c++11 -g -pedantic -Wall
PROGS   = tail tail2 wordcount wordcount-dynamic
REMOVE  = rm -f
OBJ     = hash_function.o htab_init.o htab_move.o htab_size.o htab_bucket_count.o htab_lookup_add.o htab_find.o htab_foreach.o htab_remove.o htab_clear.o htab_free.o

all: $(PROGS) 

#LINK
tail: tail.o
	$(CC) $(CFLAGS) $< -o $@
tail2: tail2.o
	$(CPP) $(CPPFLAGS) $< -o $@
wordcount: wordcount.o io.o libhtab.a
	$(CC) $(CFLAGS) -o $@ -static wordcount.o io.o -L. -lhtab
wordcount-dynamic: wordcount.o io.o libhtab.so
	$(CC) $(CFLAGS) -o $@ wordcount.o io.o -L. -lhtab

#COMPILATION
tail.o: tail.c
	$(CC) $(CFLAGS) -c $< -o $@
tail2.o: tail2.cc
	$(CPP) $(CPPFLAGS) -c $< -o $@
io.o: io.c io.h
	$(CC) $(CFLAGS) -c $< -o $@
wordcount.o: wordcount.c io.h htab.h
	$(CC) $(CFLAGS) -c $< -o $@

hash_function.o: hash_function.c htab.h
	$(CC) $(CFLAGS) -c -fPIC $< -o $@
htab_init.o: htab_init.c htab.h private.h
	$(CC) $(CFLAGS) -c -fPIC $< -o $@
htab_move.o: htab_move.c htab.h private.h
	$(CC) $(CFLAGS) -c -fPIC $< -o $@
htab_size.o: htab_size.c htab.h private.h
	$(CC) $(CFLAGS) -c -fPIC $< -o $@
htab_bucket_count.o: htab_bucket_count.c htab.h private.h
	$(CC) $(CFLAGS) -c -fPIC $< -o $@
htab_lookup_add.o: htab_lookup_add.c htab.h private.h
	$(CC) $(CFLAGS) -c -fPIC $< -o $@
htab_find.o: htab_find.c htab.h private.h
	$(CC) $(CFLAGS) -c -fPIC $< -o $@
htab_foreach.o: htab_foreach.c htab.h private.h
	$(CC) $(CFLAGS) -c -fPIC $< -o $@
htab_remove.o: htab_remove.c htab.h private.h
	$(CC) $(CFLAGS) -c -fPIC $< -o $@
htab_clear.o: htab_clear.c htab.h private.h
	$(CC) $(CFLAGS) -c -fPIC $< -o $@
htab_free.o: htab_free.c htab.h private.h
	$(CC) $(CFLAGS) -c -fPIC $< -o $@

# LIBRARIES
libhtab.a: $(OBJ)
	ar rcs $@ $^
libhtab.so: $(OBJ) 
	$(CC) $(CFLAGS) -shared -fPIC $^ -o $@ 

#CLEAN
clean: clear
clear:
	$(REMOVE) *.o $(PROGS) libhtab.a libhtab.so

