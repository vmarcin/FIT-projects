/*
 * htab_foreach.c
 * Riesenie IJC-DU2, priklad 2), 22.4.2017
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 */
#include "htab.h"
#include "private.h"

void htab_foreach(htab_t *t,void(*func)(char*,unsigned))
{
	if(t == NULL)
		return;
	
	for(unsigned i = 0; i < t->arr_size; i++)
	{
		for(struct htab_listitem *tmp= t->ptr[i]; tmp != NULL; tmp = tmp->next)
		{
			func(tmp->key, tmp->data);
		}
	}
}
