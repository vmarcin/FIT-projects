// project: Implementation and breaking of RSA
// author: Vladimir Marcin
// login: xmarci10
// mail: xmarci10@stud.fit.vutbr.cz
// date: 2.5.2020

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

#include "rsa.h"

#define DEBUG 1

void usage() {
    printf("Usage: ./kry [OPTION]\n");
    printf("\nOptions:\n");
    printf("\t-g B\t\tgenerates a RSA key B bits long\n");
    printf("\t-e E N M\tencrypts message M using exponent E and modulus N (E,N,M in hexa)\n");
    printf("\t-d D N C\tdecrypts cipher C using exponent D and modulus N (D,N,C in hexa)\n");
    printf("\t-b E N C\tbreaks the rsa cipher with exponent E and modulus N and decrypts message C (E,N,C in hexa)\n");
}

int getG_arg(int argc, char *argv[], unsigned long *modulus_size) {
    assert (argc == 3);

    char *ptr = NULL;
    *modulus_size = strtoul(argv[2], &ptr, 10);
    // rsa key length in wrong format (not a number!)
    if (*ptr != '\0') return 1;
    
    return 0;
}

int getEDB_args(int argc, char *argv[], struct rsa_class *rsa_input) {
    assert (argc == 5);

    mpz_init(rsa_input->exponent);
    mpz_init(rsa_input->modulus);
    mpz_init(rsa_input->message);

    if ( mpz_set_str(rsa_input->message, argv[4]+2, 16) ||
         mpz_set_str(rsa_input->modulus, argv[3]+2, 16) ||
         mpz_set_str(rsa_input->exponent, argv[2]+2, 16))
        return 1;

    return 0;
}

int main(int argc, char *argv[]) {
    /* PARSE INPUT */
    struct rsa_class rsa_input;
    unsigned long modulus_size;
    
    switch (getopt(argc, argv, ":gedb")) {
        case 'g':
            if ( getG_arg(argc, argv, &modulus_size) ) {
                usage();
                return 1;
            } else { rsa_gen_keys(modulus_size); }
            break;
        case 'e':
        case 'd':
            if ( getEDB_args(argc, argv, &rsa_input) ) {
                usage();
                return 1;
            } else {
                rsa_endecrypt(&rsa_input);
                mpz_clears(rsa_input.modulus, rsa_input.exponent, rsa_input.message, NULL);
            }
            break;
        case 'b':
            if ( getEDB_args(argc, argv, &rsa_input) ) {
                usage();
                return 1;
            } else {
                int retval = rsa_break(&rsa_input);
                mpz_clears(rsa_input.modulus, rsa_input.exponent, rsa_input.message, NULL);
                return retval;
            }
            break;
        case '?':
            usage();
            break;
    }

    // everything OK 
    return 0;
}
