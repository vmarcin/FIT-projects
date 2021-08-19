/*
 * htab_clear.c
 * Riesenie IJC-DU2, priklad 2), 22.4.2017
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 */
#include "htab.h"
#include "private.h"

void htab_clear(htab_t *t)
{
	if(t == NULL)
		return;
	
	struct htab_listitem *tmp = NULL;

	for(unsigned i = 0; i < t->arr_size; i++)
	{
		if(t->ptr[i] == NULL)
			continue;
	
		while(t->ptr[i] != NULL)
		{
			tmp = t->ptr[i];
			t->ptr[i] = t->ptr[i]->next;	
			if(tmp->key != NULL)
				free(tmp->key);
			free(tmp);
		}
	}
	t->n = 0;
}
