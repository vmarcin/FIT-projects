// project: Implementation and breaking of RSA
// author: Vladimir Marcin
// login: xmarci10
// mail: xmarci10@stud.fit.vutbr.cz
// date: 2.5.2020

#ifndef RSA_H
#define RSA_H

#include <gmp.h>

struct rsa_class{
    mpz_t exponent;
    mpz_t modulus;
    mpz_t message;
};

void rsa_gen_keys(unsigned long modulus_size);

void rsa_endecrypt(struct rsa_class *rsa_input);

int rsa_break(struct rsa_class *rsa_input);

#endif /* RSA_H */