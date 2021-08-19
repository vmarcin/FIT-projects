/*
 * error.h
 * Riesenie IJC-DU1, priklad b), 26.3.2016
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 *
 * Rozhranie obsahuje funkcie s premennym poctom argumentov sluziace na 
 * vypis chybovych sprav 
 */
#ifndef ERROR_H
#define ERROR_H

/* Funkcia vytlaci text "CHYBA: " a potom chybove hlasenie podla formatu fmt.
 * Vsetko sa tlaci na stderr
 */
void warning_msg(const char *fmt, ...);

/* Funkcia vytlaci text "CHYBA: " a potom chybove hlasenie podla formatu fmt.
 * Vsetko sa tlaci na stderr.
 * Funkcia ukonci program volanim exit(1)
 */
void error_msg(const char *fmt, ...);

#endif
