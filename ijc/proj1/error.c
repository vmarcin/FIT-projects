/*
 * error.c
 * Riesenie IJC-DU1, priklad b), 26.3.2016
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609 
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "error.h"

void warning_msg(const char *fmt, ...)
{
	va_list args;

	va_start(args, fmt);
	fprintf(stderr, "CHYBA: ");
	vfprintf(stderr, fmt, args);
	va_end(args);
}

void error_msg(const char *fmt, ...)
{
	va_list args;

	va_start(args, fmt);
	fprintf(stderr, "CHYBA: ");
	vfprintf(stderr, fmt, args);
	va_end(args);
	exit (1);
}
