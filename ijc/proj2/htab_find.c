/*
 * htab_find.c
 * Riesenie IJC-DU2, priklad 2), 22.4.2017
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 */
#include "htab.h"
#include "private.h"

struct htab_listitem *htab_find(htab_t *t, const char *key)
{
	if(t == NULL || key == NULL)
		return NULL;
	
	unsigned index = hash_function(key) % t->arr_size;
	struct htab_listitem *tmp = NULL;

	for(tmp = t->ptr[index]; tmp != NULL; tmp = tmp->next)
	{	
		if((strcmp(tmp->key,key)) == 0)
			return tmp;
	}
	return NULL;	
}
