/*
 * hash_function.c
 * Riesenie IJC-DU2, priklad 2), 22.4.2017
 * Autor: http://www.cse.yorku.ca/~oz/hash.html
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 */
#include "htab.h"
#include "private.h"

unsigned int hash_function(const char *str)
{
	unsigned int h = 0;
	const unsigned char *p;
	for(p = (const unsigned char*)str; *p != '\0'; p++)
		h = 65599*h + *p;
	return h;
}
