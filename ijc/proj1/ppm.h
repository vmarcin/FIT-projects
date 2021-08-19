/*
 * ppm.h
 * Riesenie IJC-DU1, priklad b), 26.3.2016
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 *
 * Rozhranie obsahuje funkcie sluziace na pracu s PPM subormi (zapis a citanie) 
 */
#ifndef PPM_H
#define PPM_H

// Typ definujuci PPM format
struct ppm {
    unsigned xsize;
    unsigned ysize;
    char data[];    // RGB bajty, celkem 3*xsize*ysize
};

/* Funkcia nacita obsah ppm suboru "filename" do dynamicky alokovanej struktury
 * a vracia ukazatel na strukturu.
 * V pripade chyby vracia NULL
 */
struct ppm * ppm_read(const char * filename);

/* Funkcia zapise obsah struktury "p" do suboru "filename".
 * V pripade chyby vrati zaporne cislo.
 */
int ppm_write(struct ppm *p, const char * filename);

#endif
