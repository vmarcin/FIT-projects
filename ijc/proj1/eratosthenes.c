/*
 * eratosthenes.c
 * Riesenie IJC-DU1, priklad a), 26.3.2016
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609 
 */
#include <stdio.h>
#include <math.h>
#include "bit_array.h"
#include "eratosthenes.h"

void Eratosthenes(bit_array_t pole)
{
	unsigned long n = ba_size(pole);

	// 0 a 1 niesu prvocisla -> nastavime na 1
	ba_set_bit(pole,0,1);
	ba_set_bit(pole,1,1);

	for(unsigned long i = 2; i <= sqrt(n); i++)
	{
		/* najdeme najmensi index taky ze hodnota bitu = 0 
		 * a vsetky jeho nasobky nastavime na 1
		 */
		if((ba_get_bit(pole,i)) == 1) continue;
		for(unsigned long j=2*i; j < n; j+=i)
		{
			ba_set_bit(pole,j,1);
		}
	}
}
