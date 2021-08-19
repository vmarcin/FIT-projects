/*
 * proj2.h
 * Riesenie IOS-proj2, 27.4.2017
 * Autor: Vladimír Marcin FIT 1-BIT (xmarci10)
 * Prelozene: gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609
 */

#ifndef PROJ2_H
#define PROJ2_H

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <semaphore.h>
#include <time.h>
#include <sys/mman.h>

#define semMUTEX            "/xmarci10_mutex"
#define semCHILD_QUEUE      "/xmarci10_childQueue"
#define semADULT_QUEUE      "/xmarci10_adultQueue"
#define semEND							"/xmarci10_end"
#define	semCHILD_ENTER			"/xmarci10_childEnter"
#define semADULT_LEAVE			"/xmarci10_adultLeave"

#define shmSIZE             sizeof(int)
#define shmCHILDREN         "/xmarci10_shmChildren"
#define shmADULTS           "/xmarci10_shmAdults"
#define shmWAITING          "/xmarci10_shmWaiting"
#define shmLEAVING          "/xmarci10_shmLeaving"
#define shmCOUNTER          "/xmarci10_shmCounter"
#define shmADULTS_COMING    "/xmarci10_shmAdultsComing"
#define shmPROCESS_COUNTER	"/xmarci10_shmProcessCounter"

/*
 * struktura ktora obsahuje argumenty programu
 *	-> adults 	= pocet procesov adult
 *	-> children = pocet procesov child
 *	-> agt			= max. doba po ktorej je generovany novy proces adult
 *	-> cgt			= max. doba po ktorej je generovany novy proces child
 *	-> awt			= max. doba pocas ktorej proces adult simuluje svoju cinnost
 *	-> cwt			= max. doba pocas ktorej proces child simuluje svoju cinnost
 */
typedef struct{					
  int adults;
  int children;
  int agt;
  int cgt;
  int awt;
  int cwt;
}arguments;


/*
 * Funkcia spracuje argumenty z prikazoveho riadku
 * a vracia strukturu ktora obsahuje hodnoty vsetkych
 * argumentov.
 * V pripade chyby vypise hlasku na stderr a ukonci program
 */
arguments test_argument(int argc, char *argv[]);

/*
 * Funkcia v pripade zle zadanych argumentov vypise
 * na stderr spravne pouzitie programu
 */
void usage();

/*
 * Funkcia riesi synchronizaciu procesu adult pomocou
 * semaforov. Parameter awt urcuje maximalnu dobu pocas
 * ktorej proces simuluje svoju cinnost a parameter
 * id identifikuje dany proces. 
 */
void adult(int awt, int id);

/*
 * Funkcia riesi synchronizaciu procesu child pomocou
 * semaforov. Parameter awt urcuje maximalnu dobu pocas
 * ktorej proces simuluje svoju cinnost a parameter
 * id identifikuje dany proces. 
 */
void child(int cwt, int id);

/*
 * Funkcia vytvori semafory a v pripade chyby
 * vypise chybu na stderr a ukonci program
 */
void create_semaphores();

/*
 * Funkcia zatvori a odstrani vsetky vytvorene semafory
 * v pripade neuspechu vypise varovanie na stderr a ukonci
 * program
 */
void free_semaphores();

/*
 * Funkcia vytvori zdielanu pamat a v pripade ze vsetko
 * prebehlo uspesne vrati 'true' v pripade chyby vracia 'false'
 */
bool create_shm(arguments arg);

/*
 * Funkcia uvolni zdielanu pamet a v pripade chyby 
 * vypise chybu na stderr a ukonci program
 */
void free_shm();

/*
 * Funkcia vrati vygenerovane cislo z intervalu
 * <0,max>. Ak je hodnota max 0 vrati 0	
 */
unsigned random_number(int max);
#endif
