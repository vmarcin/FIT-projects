/*
 * steg-decode.c
 * Riesenie IJC-DU1, priklad b), 26.3.2016
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609 
 */
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <limits.h>
#include <ctype.h>

#include "eratosthenes.h"
#include "error.h"
#include "bit_array.h"
#include "ppm.h"

#define MAX 1000*1000*3		//max. velkost obrazovych dat

int main(int argc, char *argv[])
{
	if(argc != 2)
	{
		error_msg("Zly pocet argumentov!\n");
	}

	struct ppm *ppm_image = ppm_read(argv[1]);
	//ppm_write(ppm_image,"my.ppm");
	
	if (ppm_image == NULL)
	{
		error_msg("Chyba pri praci so suborom!\n");
	}
	
	//vytvorenie bitoveho pola a najdenie prvocisel	
	ba_create(primes, MAX);	
	Eratosthenes(primes);

	char c = 0;		//dekodovany znak spravy
	char lsb_data;	//hodnota najnizsieho bitu vybraneho bajtu(podla prvocisla)
	int offset = 0; //cislo bitu v ramci spracovavaneho bajtu "c"
	bool end = 1;	//kontrola posledneho znaku '\0'
	int i;

	for(i = 2; i < MAX -1; i++)
	{
		//najdeme prvocislo
		if(!(ba_get_bit(primes, i)))
		{
			lsb_data = ppm_image->data[i] & 1;	//v lsb_data mame ulozeny posledny bit daneho bajtu
			c |= (lsb_data << offset);			//k hodnote c pridame bit lsb_data
			offset += 1;
			
			//ak sme nacitali cely bajt zistujeme ci nieje koncovy alebo netlacitelny znak ak nieje citame dalej
			if(offset == CHAR_BIT)
			{
				if(c == 0)
				{
					end = 0;
					break;
				}
				if(!isprint(c))
				{
					free(ppm_image);
					error_msg("Sprava obsahuje netlacitelny znak!\n");
				}
				printf("%c",c);
				offset = 0;
				c = 0;
			}
		}
	}
	printf("\n");
	free(ppm_image);
	if(end)
	{
		error_msg("Nekorektne ukoncena sprava!\n");
	}
	return 0;		
}

