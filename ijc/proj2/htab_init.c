/*
 * htab_init.c
 * Riesenie IJC-DU2, priklad 2), 22.4.2017
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 */
#include "htab.h"
#include "private.h"

htab_t *htab_init(unsigned size)
{
	htab_t *t = (htab_t *)malloc(sizeof(htab_t) + size * sizeof(struct htab_listitem *));
	if(t == NULL)
		return NULL;
	t->arr_size = size;	
	t->n = 0;
	for(unsigned i = 0; i < t->arr_size; i++)
		t->ptr[i] = NULL;
	return t;
}
