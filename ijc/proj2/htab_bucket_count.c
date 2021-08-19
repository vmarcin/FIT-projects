/*
 * htab_bucket_count.c
 * Riesenie IJC-DU2, priklad 2), 22.4.2017
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 */
#include "htab.h"
#include "private.h"

size_t htab_bucket_count(htab_t *t)
{
	if(t == NULL)
		return -1;
	return t->arr_size;
}
