/*
 * io.h
 * Riesenie IJC-DU2, priklad 2), 22.4.2017
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 */
#ifndef IO_H
#define	IO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>

//premenna indikujuca presiahnutie maximalnej velkosti slova
extern bool err;

/*
 * funkcia nacita jedno slovo do 's' a vrati jeho dlzku
 * v pripade prekrocenia limitu vrati hodnotu max-1
 * ak dosiadne koniec suboru vrati EOF 
*/
int get_word(char *s, int max, FILE *f);

#endif
