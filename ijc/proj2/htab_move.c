/*
 * htab_move.c
 * Riesenie IJC-DU2, priklad 2), 22.4.2017
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 */
#include "htab.h"
#include "private.h"

htab_t *htab_move(unsigned newsize, htab_t *t2)
{
	if(t2 == NULL)
		return NULL;
	htab_t *t = htab_init(newsize);
	if(t == NULL)
		return NULL;
	
	struct htab_listitem *tmp = NULL;
	struct htab_listitem *nextItem = NULL;
	unsigned index;

	for(unsigned i = 0; i < t2->arr_size; i++)
	{
		if(t2->ptr[i] == NULL)
			continue;
		for(tmp = t2->ptr[i]; tmp != NULL; tmp = nextItem)
		{
			index = hash_function(tmp->key) % t->arr_size;
			nextItem = tmp->next;
			if(t->ptr[index] == NULL)
			{
				t->ptr[index] = tmp;
				t->ptr[index]->next = NULL;
			}
			else
			{
				tmp->next = t->ptr[index];
				t->ptr[index] = tmp;
			}
			t->n++;
			t2->n--;
		}
		t2->ptr[i] = NULL;
	}	
	return t;
}
