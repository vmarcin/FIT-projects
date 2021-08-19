/*
 * ppm.c
 * Riesenie IJC-DU1, priklad b), 26.3.2016
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609 
 */
#include <stdio.h>
#include <stdlib.h>
#include "ppm.h"
#include "error.h"

struct ppm * ppm_read(const char * filename)
{
	FILE *fp = fopen(filename,"r");
	struct ppm *ppm_image = NULL;
	unsigned long size; // urcuje velkost dat v PPM subore
	
	unsigned rgb; // max. hodnota farby v PPM subore
	unsigned xsize;
	unsigned ysize;
	char type[16]; // typ PPM suboru
	
	if(fp == NULL)
	{
		warning_msg("Nepodarilo sa otvorit subor %s!\n", filename);
		return NULL;
	}
	
	//nacitanie typu PPM suboru
	if((fgets(type, sizeof(type), fp) == NULL))
	{
		fclose(fp);
		warning_msg("Neuspesne citanie zo suboru %s!\n", filename);
		return NULL;
	}

	//kontrola typu PPM suboru
	if(type[0] != 'P' || type[1] != '6')
	{
		warning_msg("Chybny format suboru %s!\n", filename);
		fclose(fp);
		return NULL;
	}
	
	//nacitanie xsize ysize a farby
	if((fscanf(fp, "%u %u %u ", &xsize, &ysize, &rgb)) != 3)
	{
		warning_msg("Neuspesne citanie zo suboru %s!\n", filename);
		fclose(fp);
		return NULL;			
	}
	if(rgb != 255)
	{
		warning_msg("Zly rozsah farieb v subore %s!\n", filename);
		fclose(fp);
		return NULL;
	}	

	size = xsize * ysize * 3 * sizeof(char);
	
	//alokacia struktury
	if((ppm_image = (struct ppm *)malloc(sizeof(struct ppm) + size)) == NULL)
	{
		warning_msg("Nepodarila sa alokacia struktury!\n");
		fclose(fp);
		return NULL;
	}

	ppm_image->xsize = xsize;
	ppm_image->ysize = ysize;
	
	//citanie dat z PPM suboru
	if((fread(ppm_image->data, sizeof(char), size, fp)) != size || fgetc(fp) != EOF)
	{
		warning_msg("Neuspesne citanie zo suboru %s!\n", filename);
		free(ppm_image);
		fclose(fp);
		return NULL;
	}
	fclose(fp);	
	return ppm_image;
}


int ppm_write(struct ppm *p, const char * filename)
{
	FILE *fp = fopen(filename, "wb");
	unsigned long size = p->xsize * p->ysize * sizeof(char) * 3;

	if(fp == NULL)
	{
		warning_msg("Nepodarilo sa otvorit subor %s!\n", filename);
		return -1;
	}

	//zapis parametrov PPM
	if((fprintf(fp,"P6\n%u %u\n255", p->xsize, p->ysize)) == 0)
	{
		warning_msg("Chyba pri zapisovani do suboru %s!\n", filename);
		fclose(fp);
		return -1;
	}

	//zapis dat do PPM suboru
	if((fwrite(p->data, sizeof(char), size, fp)) != size)
	{
		warning_msg("Chyba pri zapisovani do suboru %s!\n", filename);
		fclose(fp);
		return -1;
	}
	fclose(fp);
	return 0;
}
