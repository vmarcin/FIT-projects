/*
 * wordcount.c
 * Riesenie IJC-DU2, priklad 2), 22.4.2017
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "htab.h"
#include "io.h"

#define MAX_WORD	  127	
/*
 * Pri vybere vhodnej velkosti hashovacej tabulky som sa inspiroval hodnotami na stranke:
 * http://www.orcca.on.ca/~yxie/courses/cs2210b-2011/htmls/extra/PlanetMath_%20goodhashtable.pdf
 * Zvolil som hodnotu "12289" na zaklade nizkej chybovosti v porovnani so susednymi hodnotami
 * a uspokojivymi vysledkami pri testovani. V testoch som pouzil subory ktore obsahovali
 * priblizne 10 000 unikatnych slov. 
 */
#define SIZE_HTABLE 12289

bool err = false;

void htab_print(char *key, unsigned data);

int main(void)
{
	char word[MAX_WORD];	
	int lenght;
	struct htab_listitem *item = NULL;
	htab_t *t = htab_init(SIZE_HTABLE);
		
	if(t == NULL)
	{
		fprintf(stderr,"CHYBA: Chyba pri inicializacii tabulky!\n");
		return 1;
	}
	
	while((lenght = get_word(word,MAX_WORD,stdin)) != EOF)
	{
		item = htab_lookup_add(t,word);
		if(item == NULL)
		{
			fprintf(stderr,"CHYBA: Chyba pri pridavani prvku do tabulky!\n");
			htab_free(t);
			return 1;
		}
	}
	//ak sa EOF nenachadza na novom riadku potrebujeme nacitat este jedno slovo	
	if(word[0] != 0)
	{
		item = htab_lookup_add(t,word);
		if(item == NULL)
		{
			fprintf(stderr,"CHYBA: Chyba pri pridavani prvku do tabulky!\n");
			htab_free(t);
			return 1;
		}
	}
	htab_foreach(t,htab_print);
	htab_free(t);
	
	if(err)
		fprintf(stderr,"VAROVANIE: Niektore slovo prekrocilo max limit a bolo skratene!\n");
	return 0;
}

void htab_print(char *key, unsigned data)
{
	printf("%s\t%u\n",key,data);
}
