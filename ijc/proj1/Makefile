CC= gcc
CFLAGS= -O2 -std=c99 -Wall -pedantic
LIBS= -lm
INLINE= -DUSE_INLINE
REMOVE= rm -f

all: primes primes-i steg-decode
# linkovanie
primes: primes.o eratosthenes.o error.o
		$(CC) $(CFLAGS) primes.o eratosthenes.o error.o -o primes $(LIBS)
primes-i: primes-i.o eratosthenes-i.o error.o
		$(CC) $(CFLAGS) primes-i.o eratosthenes-i.o error.o -o primes-i $(LIBS)
steg-decode: eratosthenes.o steg-decode.o ppm.o error.o
		$(CC) $(CFLAGS) eratosthenes.o steg-decode.o ppm.o error.o -o steg-decode $(LIBS)

# kompilacia
primes.o: primes.c bit_array.h eratosthenes.h
		$(CC) $(CFLAGS) -c primes.c -o primes.o
primes-i.o: primes.c bit_array.h eratosthenes.h
		$(CC) $(CFLAGS) $(INLINE) -c primes.c -o primes-i.o

eratosthenes.o: eratosthenes.c bit_array.h eratosthenes.h
		$(CC) $(CFLAGS) -c eratosthenes.c -o eratosthenes.o
eratosthenes-i.o: eratosthenes.c bit_array.h eratosthenes.h
		$(CC) $(CFLAGS) $(INLINE) -c eratosthenes.c -o eratosthenes-i.o
error.o: error.c error.h
		$(CC) $(CFLAGS) -c error.c -o error.o

steg-decode.o: steg-decode.c ppm.h error.h bit_array.h eratosthenes.h
		$(CC) $(CFLAGS) -c steg-decode.c -o steg-decode.o
ppm.o: ppm.c ppm.h error.h
		$(CC) $(CFLAGS) -c ppm.c -o ppm.o

# spustenie
run:
	time ./primes
	time ./primes-i

# vycistenie souborov prekladu
clean: clear
clear:
	$(REMOVE) *.o primes primes-i steg-decode
