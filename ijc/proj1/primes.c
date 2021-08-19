/*
 * primes.c
 * Riesenie IJC-DU1, priklad a), 26.3.2016
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609 
 */

#include <stdio.h>
#include "bit_array.h"
#include "eratosthenes.h"

#define N 303000000

int main(int argc, char *argv[])
{

	//vytvorenie bitoveho pola
	ba_create(pole,N);
	//vypocet prvocisel
	Eratosthenes(pole);
    
	unsigned long primes[10] = {0};
	int j = 0;
	for(int i = N-1; i >= 0 && j < 10; i--)
	{
		if((ba_get_bit(pole,i)) == 0)
		{
			primes[j] = i;
			j+=1;
		}
	}

	for(int i = 9; i >= 0; i--)
	{
		printf("%lu\n",primes[i]);
	}
}
