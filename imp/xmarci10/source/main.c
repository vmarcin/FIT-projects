/*
 * main.c (original)
 *
 *  Created on: Dec 19, 2018
 *      Author: Vladimír Marcin (xmarci10@stud.fit.vutbr.cz)
 */


/* Libraries included */
#include "board.h"
#include "pin_mux.h"
#include "clock_config.h"
#include "MK60D10.h"

#include "fsl_clock.h"
#include "fsl_uart.h"
#include "fsl_device_registers.h"
#include "fsl_debug_console.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Own libraries */
#include "crc.h"

#define CAPACITY 128

/* CRC Tables */
uint16_t crcTable16[TAB_SIZE];
uint32_t crcTable32[TAB_SIZE];

/**
 * Function prints 'prompt' string and reads data given
 * by user and saves it to 'input' string
 */
void getInputData( char *input, const char *prompt );

/**
 * Function just print ASCII art line (just for formating)
 */
inline void printLine();

/**
 * Function get results of all variants and print it
 */
void getResults( uint16_t poly16, uint32_t poly32, char *data);

/**
 * Get dialog answer from user
 */
char getUsersAnswer( const char *prompt );

/**
 * Get polynoms values from users and set it to 'poly16' and 'poly32'
 */
void setPolynomials( uint16_t *poly16, uint32_t *poly32, char *input );

int main( void ) {
    /* Init board hardware. */
    BOARD_InitPins();
    BOARD_BootClockRUN();
    BOARD_InitDebugConsole();

    uint16_t poly16;
    uint32_t poly32;
    char *input = ( char * ) malloc(CAPACITY);

    PRINTF ("ZABEZPECENIE DAT POMOCOU 16/32-BIT. KODU CRC\n\r");

    while (true) {
    	setPolynomials( &poly16, &poly32, input );

		getInputData( input, "Zadajte retazec, ktory chcete zabezpecit: " );
		getResults(poly16, poly32, input);

		getInputData( input, "Zadajte (\"prijaty\") retazec na overenie: " );
		getResults(poly16, poly32, input);

    	char usersAnswer = getUsersAnswer("Chcete zadat dalsi retazec? y/n: ");
		if ( usersAnswer == 'n' ) {
			printLine();
			PRINTF("DOVIDENIA! :)");
			free(input);
			break;
		} else {
			printLine();
		}
    }
}

void getInputData( char *input, const char *prompt ) {
	PRINTF("\n\r%s", prompt);

    char c = GETCHAR();
    PUTCHAR( c );

    for ( int i = 0; c != '\r'; ) {
    	if ( i > CAPACITY )
			input = ( char* )realloc( input, 2*i );

		if ( c == 127 ) {
			input[--i] = '\0';
		} else {
			input[i] = c;
			input[++i] = '\0';
		}
		c = GETCHAR();
		PUTCHAR( c );
    }
}

void printLine() {
	PRINTF("\n\r+------------------------------------------------------------+\n\r");
}

void getResults( uint16_t poly16, uint32_t poly32, char *data) {
	printLine();
	PRINTF("CRC16\r\n\n");

	PRINTF("HW MODUL:\t0x%04x\n\r", computeCRC16HWMOD( data, poly16 ));
	calculateTableCRC16(poly16);
	PRINTF("TABULKA:\t0x%04x\n\r", computeCRC16Table( data ) );
	PRINTF("POLYNOM:\t0x%04x", computeCRC16Simple( data, poly16 ));

	printLine();
	PRINTF("CRC32\r\n\n");

	PRINTF("HW MODUL:\t0x%08x\n\r", computeCRC32HWMOD( data, poly32 ));
	calculateTableCRC32(poly32);
	PRINTF("TABULKA:\t0x%08x\n\r", computeCRC32Table( data ) );
	PRINTF("POLYNOM:\t0x%08x", computeCRC32Simple( data, poly32 ));
	printLine();
}

char getUsersAnswer( const char *prompt ) {
	char ans;

	PRINTF("\n\r%s", prompt);
	while ( ( ans = GETCHAR() ) != 'y' && ( ans != 'n') );

	return ans;
}

void setPolynomials( uint16_t *poly16, uint32_t *poly32, char *input ) {
	char usersAnswer = getUsersAnswer("Chcete pouzit predvoleny polynom pre CRC16 (\"0x1021\")? y/n: ");
	if ( usersAnswer == 'y' ) {
		*poly16 = 0x1021U;
	} else {
		getInputData( input, "Zadajte polynom, ktory chete pouzit na zabezpecenie dat pomocou CRC16: 0x" );
		*poly16 = strtol( input, NULL, 16);
	}

	usersAnswer = getUsersAnswer("Chcete pouzit predvoleny polynom pre CRC32 (\"0x4C11DB7\")? y/n: ");
	if ( usersAnswer == 'y' ) {
		*poly32 = 0x4C11DB7U;
	} else {
		getInputData( input, "Zadajte polynom, ktory chete pouzit na zabezpecenie dat pomocou CRC32: 0x" );
		*poly32 = strtol( input, NULL, 16);
	}
	printLine();
}
