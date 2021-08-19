/*
 * htab_lookup_add.c
 * Riesenie IJC-DU2, priklad 2), 22.4.2017
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 */
#include "htab.h"
#include "private.h"

struct htab_listitem *htab_lookup_add(htab_t *t, const char *key)
{
	if(t == NULL || key == NULL)
		return NULL;
	
	struct htab_listitem *tmp = htab_find(t,key);
	if(tmp != NULL)
	{
		tmp->data++;
		return tmp;
	}
	
	unsigned index = hash_function(key) % t->arr_size;
	struct htab_listitem *newItem = (struct htab_listitem *)malloc(sizeof(struct htab_listitem));
	if(newItem == NULL)
		return NULL;

	newItem->key = malloc((strlen(key)+1) * sizeof(char));
	if(newItem->key == NULL)
		return NULL;

	strcpy(newItem->key,key);
	newItem->data = 1;
	newItem->next = NULL;

	//vlozenie zaznamu na zaciatok zoznamu	
	tmp = t->ptr[index];
	newItem->next = tmp;
	t->ptr[index] = newItem;
	
	t->n++;
		
	return newItem;			
}
