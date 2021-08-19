/*
 * bit_array.h
 * Riesenie IJC-DU1, priklad a), 26.3.2016
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 * 
 * Rozhranie obsahuje makra pre pracu s "bitovym polom".
 * Podmienenim prekladom zaistime aby sa namiesto makier definovali inline funckie
 * s rovnakym menom.
 */
#ifndef BIT_ARRAY_H
#define BIT_ARRAY_H

#include <stdio.h>
#include <limits.h>
#include <stdbool.h>
#include "error.h"

typedef unsigned long *bit_array_t;

//vrati pocet bitov v unsigned long
#define BITS_IN_UL (CHAR_BIT * sizeof(unsigned long))

//vrati hodnotu bitu na zadanom indexe "i" v poli "p"
#define BA_GET_BIT_(p,i)\
	(p[(i) / (sizeof(p[0])*8)] & (1LU << (i % (sizeof(p[0])*8) ))) ? 1 : 0

//nastavi hodnotu bitu na indexe "i" v poli "p" podla hodnoty vyrazu "b"
#define BA_SET_BIT_(p,i,b)\
	((b))?\
	(p[(i) / (sizeof(p[0])*8)] |= (1LU << (i % (sizeof(p[0])*8)))) :\
	(p[(i) / (sizeof(p[0])*8)] &= ~(1LU << (i % (sizeof(p[0])*8))))	

//vrati velkost potrebnu na vytvorenie pola
#define BIT_ARRAY_SIZE(velikost)\
	(velikost % BITS_IN_UL ) ?\
	(velikost / BITS_IN_UL + 2) :\
	(velikost / BITS_IN_UL + 1)

//definuje a nuluje pole "jmeno_pole" 
#define ba_create(jmeno_pole,velikost)\
	unsigned long jmeno_pole [BIT_ARRAY_SIZE(velikost)] = {(velikost),0}

/*                          DEFINICIA MAKIER                                 */
#ifndef USE_INLINE

//vrati deklarovanu velkost pola v bitoch
#define ba_size(jmeno_pole)\
	jmeno_pole[0]

/* nastavi hodnotu bitu na indexe "index" v poli "jmeno_pole" podla hodnoty vyrazu "vyraz"
 * pozn.: kontroluje medze poli
 */
#define ba_set_bit(jmeno_pole,index,vyraz)\
	(index >= ba_size((jmeno_pole))) ?\
	error_msg("Index %lu mimo rozsah 0..%ld",(unsigned long)index,(unsigned long)ba_size(jmeno_pole)-1),4 :\
	BA_SET_BIT_((jmeno_pole),(index+BITS_IN_UL),(vyraz))

/* vrati hodnotu bit na indexe "index" v poli "jmeno_pole"
 * pozn.: kontroluje medze poli
 */
#define ba_get_bit(jmeno_pole,index)\
	(index >= ba_size((jmeno_pole))) ?\
	error_msg("Index %lu mimo rozsah 0..%ld",(unsigned long)index,(unsigned long)ba_size(jmeno_pole)-1),4 :\
	BA_GET_BIT_((jmeno_pole),(index+BITS_IN_UL))
#endif

/*                       DEFINICIA INLINE FUNKCII                            */
#ifdef USE_INLINE

//vrati deklarovanu velkost pola v bitoch
inline unsigned long ba_size(bit_array_t jmeno_pole)
{
	return jmeno_pole[0];
}

/* vrati hodnotu bit na indexe "index" v poli "jmeno_pole"
 * pozn.: kontroluje medze poli
 */
inline bool ba_get_bit(bit_array_t jmeno_pole, unsigned long index)
{
	if(index >= ba_size(jmeno_pole))
		error_msg("Index %lu mimo rozsah 0..%ld",(unsigned long)index,(unsigned long)ba_size(jmeno_pole)-1);
	return (BA_GET_BIT_((jmeno_pole),(index+BITS_IN_UL)));	
}

/* nastavi hodnotu bitu na indexe "index" v poli "jmeno_pole" podla hodnoty vyrazu "vyraz"
 * pozn.: kontroluje medze poli
 */
inline void ba_set_bit(bit_array_t jmeno_pole, unsigned long index, unsigned long vyraz)
{
	if(index >= ba_size(jmeno_pole))
		error_msg("Index %lu mimo rozsah 0..%ld",(unsigned long)index,(unsigned long)ba_size(jmeno_pole)-1);
	BA_SET_BIT_((jmeno_pole),(index+BITS_IN_UL),(vyraz));	
}
#endif

#endif
