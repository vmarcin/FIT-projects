/*
 * io.c
 * Riesenie IJC-DU2, priklad 2), 22.4.2017
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 */
#include "io.h"

int get_word(char *s, int max, FILE *f)
{
	max--;
	s[0] = 0; //nastavime prvy prvok pola na 'null' ak sa v poli 's' nachadza nacitane slovo
  char c = getc(f);
  while(isspace(c))
    c = getc(f);
	int i;
  for(i = 0; c != EOF && !isspace(c); i++, c = getc(f))
  {
    if(i < max)
    {   
      s[i] = c;
      s[i+1] = '\0';
    }
		else
			err = true;   
  }
  if(i <= max && c != EOF)
    return i;
  if(i > max && c != EOF)
    return max;
  return EOF;	
}
