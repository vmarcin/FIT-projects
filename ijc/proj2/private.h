/*
 * private.h
 * Riesenie IJC-DU2, priklad 2), 22.4.2017
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 *
 * Rozhranie obsahuje definiciu struktury hashovacej tabulky
 * ktora je privatna pre kniznicu
 */

#ifndef PRIVATE_H
#define PRIVATE_H

#include <stdio.h>
#include "htab.h"

struct htab_t{
	unsigned 	arr_size;		//velkost pola ukazatelov
	unsigned	n;					//aktualni pocet zaznamu
	struct 		htab_listitem *ptr[];	//pole ukazatelov na zaznamy
};

#endif
