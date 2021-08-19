/*
 * htab_remove.c
 * Riesenie IJC-DU2, priklad 2), 22.4.2017
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 */
#include "htab.h"
#include "private.h"

bool htab_remove(htab_t *t, const char *key)
{
	if(t == NULL || key == NULL)
		return false;

	unsigned index = hash_function(key) % htab_bucket_count(t);
	struct htab_listitem *tmp = NULL;
	struct htab_listitem *prev= NULL;	

	for(tmp = t->ptr[index]; tmp != NULL; prev = tmp, tmp = tmp->next)
	{
		if((strcmp(key,tmp->key)) == 0)
		{
			if(tmp == t->ptr[index])
				t->ptr[index] = tmp->next;
			else
				prev->next = tmp->next;

			free(tmp->key);
			free(tmp);
			t->n--;
			return true;
		}		
	}
	return false;
}
