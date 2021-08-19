/*
 * htab.h
 * Riesenie IJC-DU2, priklad 2), 22.4.2017
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 *
 * Rozhranie obsahuje funkcie pre pracu s hashovacou tabulkou
 */
#ifndef H_TABLE
#define H_TABLE

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

struct htab_listitem{
	char 			*key;			//ukazatel na dynamicky alokovany retazec
	unsigned	data;			//pocet vyskytov
	struct 		htab_listitem *next; //ukazatel na dalsi zaznam
};

//neuplna deklaracia struktury popisujucej hashovaciu tabulku
typedef struct htab_t htab_t;

/*
 * Rozptylovacia funckia pre retazce 
 * (http://www.cse.yorku.ca/~oz/hash.html varianta sdbm)
 * Funkcia vracia index do tabulky.
 */
unsigned int hash_function(const char *str);

/*
 * Funkcia vytvori a inicializuje tabulku.
 * Vrati ukazatel na tabulku v pripade neuspechu vrati NULL.
 */
htab_t *htab_init(unsigned size);

/*
 * Funkcia vytvori a inicializuje tabulku z tabulky t2.
 * Tabulka t2 ostane prazdna a alokovana.
 * Funckia vrati ukazatel na tabulku v pripade chyby NULL. 
 */
htab_t *htab_move(unsigned newsize, htab_t *t2);

/*
 * Funkcia vrati pocet prvkov tabulky. (n)
 * v pripade chyby vrati -1
 */
size_t htab_size(htab_t *t);

/*
 * Funkcia vrati pocet prvkov pola ukazatelov. (arr_size)
 * v pripade chyby vrati -1
 */
size_t htab_bucket_count(htab_t *t);

/*
 * Funckia v tabulke t vyhlada zaznam odpovedajuci retazcu key
 * a ak ho najde vrati ukazatel nan ak nie tak prvok key
 * sa prida do tabulky a vrati ukazatel nan.
 * V pripade chyby vracia NULL.
 */
struct htab_listitem *htab_lookup_add(htab_t *t, const char *key);

/*
 * Funkcia v tabulke t vyhlada zaznam v pripade uspechu vrati
 * ukazatel nan v pripade neuspechu vrati NULL.
 */
struct htab_listitem *htab_find(htab_t *t, const char *key);

/*
 * Funkcia vola zadanu funkciu pre kazdy prvok tabulky.
 * (nemeni obsah tabulky)
 */
void htab_foreach(htab_t *t,void(*func)(char*, unsigned));

/*
 * Funkcia vyhlada a zrusi polozku key v tabulke t.
 * V pripade ze sa polozka v tabulke nenachadza vrati 'false'.
 */
bool htab_remove(htab_t *t, const char *key);

/*
 * Funkcia zrusi vsetky polozky v tabulke t.
 * (tabulka ostane prazdna)
 */
void htab_clear(htab_t *t);

/*
 * Funkcia zrusi tabulku.
 */
void htab_free(htab_t *t);

#endif
