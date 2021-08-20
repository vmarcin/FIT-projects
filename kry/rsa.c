// project: Implementation and breaking of RSA
// author: Vladimir Marcin
// login: xmarci10
// mail: xmarci10@stud.fit.vutbr.cz
// date: 2.5.2020

#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "rsa.h"
#include "primes.h"

/* GENERATE RSA */

unsigned urandom(void) {
    unsigned r;
    FILE *fp = fopen("/dev/urandom", "r");
    fread((void *)&r, sizeof r, 1, fp);
    fclose(fp);
    return r;
}

void factor2(mpz_t n, mpz_t d, unsigned long *r) {
    mpz_set(d, n);

    if(mpz_odd_p(d)) 
        mpz_sub_ui(d, d, 1);
    
    *r = mpz_scan1(d, *(mp_bitcnt_t[]){0});
    mpz_fdiv_q_2exp(d, d, *r);
}

int miller_rabin(mpz_t n, unsigned long k) {
    if (mpz_cmp_ui(n, 3) <= 0 || mpz_even_p(n))
        return mpz_cmp_ui(n, 3) == 0 || mpz_cmp_ui(n, 2) == 0;

    gmp_randstate_t rs;
    gmp_randinit_default(rs);
    gmp_randseed_ui(rs, urandom());

    /* write n as 2r·d + 1 with d odd (by factoring out powers of 2 from n − 1) */
    mpz_t d, a, x;
    unsigned long r;
    mpz_init(d);
    mpz_init(a);
    mpz_init(x);
    factor2(n, d, &r);

    mpz_t n_minus_3;
    mpz_init_set(n_minus_3, n);
    mpz_sub_ui(n_minus_3, n_minus_3, 3);

    mpz_t n_minus_1;
    mpz_init_set(n_minus_1, n);
    mpz_sub_ui(n_minus_1, n_minus_1, 1);

    int is_prime = 1;

    for (unsigned long i = 0; i < k; i++) {
        /* pick a random integer a in the range [2, n − 2] */
        mpz_urandomm(a, rs, n_minus_3);
        mpz_add_ui(a, a, 2);
        /* x ← a^d mod n */
        mpz_powm(x, a, d, n);

        if (mpz_cmp_ui(x, 1) == 0 || mpz_cmp(x, n_minus_1) == 0)
            continue;

        is_prime = 0;
        for (unsigned long j = 0; j < (r-1); j++) {
            /* x ← x^2 mod n */
            mpz_powm_ui(x, x, 2, n);
            if (mpz_cmp_ui(x, 1) == 0)
                break;
            if (mpz_cmp(x, n_minus_1) == 0) {
                is_prime = 1;
                break;
            }
        }
        if (is_prime == 0)
            break;
    }

    gmp_randclear(rs);
    mpz_clears(x, a, d, n_minus_1, n_minus_3, NULL);

    return is_prime;
}

void gcd(mpz_t z, const mpz_t x, const mpz_t y) {
    mpz_t a; mpz_init_set(a, x);
    mpz_t b; mpz_init_set(b, y);
    mpz_t c; mpz_init(c);

    while(mpz_cmp_ui(a, 0) != 0) {
        mpz_set(c, a);
        mpz_mod(a, b, a);
        mpz_set(b, c);
    }
    /* return b*/
    mpz_set(z,b);

    mpz_clears(a,b,c,NULL);
}

void invert(mpz_t i, const mpz_t k, const mpz_t l) {
    mpz_t a; mpz_init_set(a, k);
    mpz_t b; mpz_init_set(b, l);
    mpz_t x; mpz_init_set_ui(x, 0);
    mpz_t y; mpz_init_set_ui(y, 1);
    mpz_t u; mpz_init_set_ui(u, 1);
    mpz_t v; mpz_init_set_ui(v, 0);
    mpz_t m; mpz_init(m);
    mpz_t n; mpz_init(n);
    mpz_t q; mpz_init(q);
    mpz_t r; mpz_init(r);


    while(mpz_cmp_ui(a, 0) != 0) {
        mpz_tdiv_q(q, b, a); mpz_tdiv_r(r, b, a);
        mpz_mul(m, u, q); mpz_sub(m, x, m);
        mpz_mul(n, v, q); mpz_sub(n, y, n);
        mpz_set(b, a); mpz_set(a, r); mpz_set(x, u);
        mpz_set(y, v); mpz_set(u, m); mpz_set(v, n);
    }
    /* return x */
    mpz_set(i, x);

    mpz_clears(a, b, x, y, u, v, m, n, q, r, NULL);
}

void rsa_gen_keys(unsigned long modulus_size) {
    assert(modulus_size > 4);

    mpz_t p; mpz_init(p);
    mpz_t q; mpz_init(q);
    mpz_t n; mpz_init(n);
    mpz_t phi; mpz_init(phi);
    mpz_t tmp1; mpz_init(tmp1);
    mpz_t tmp2; mpz_init(tmp2);
    mpz_t e; mpz_init_set_ui(e, 3);
    mpz_t d; mpz_init(d);

    unsigned long p_bits = ceil(modulus_size/2.0);
    unsigned long q_bits = floor(modulus_size/2.0);

    unsigned long p_bytes = ceil(p_bits/8.0);
    unsigned long q_bytes = ceil(q_bits/8.0);

    // ensure that modulus will have max 'modulus_size' bits
    unsigned p_mask_MSB =  (1 << p_bits - ((p_bytes - 1) * 8));    
    unsigned q_mask_MSB = (1 << q_bits - ((q_bytes - 1) * 8));

    char *pbuf = calloc(p_bytes, sizeof(char)); 
    char *qbuf = calloc(q_bytes, sizeof(char));

    /* Select p and q */
    /* Start with p */
    srand(urandom());
    for(unsigned i = 0; i < p_bytes; i++) {
        pbuf[i] = rand() % 0xFF;
    } 
    // set the bottom bit to 1 to ensure p is odd (better for finding primes)
    pbuf[p_bytes-1] |= 0x01;
    // excess bits are removed 
    pbuf[0] %= p_mask_MSB;
    // set the top (MSB) bit to 1 to ensure that 'p' is relatively large
    pbuf[0] |= (p_mask_MSB >> 1);

    mpz_import(p, p_bytes, 1, sizeof(pbuf[0]), 0, 0, pbuf);
    /* next prime */
    while(miller_rabin(p, 64) == 0) {mpz_add_ui(p, p, 2);}
    free(pbuf);

    /* Now select q */
    do { 
        srand(urandom());
        for(unsigned i = 0; i < q_bytes; i++) {
            qbuf[i] = rand() % 0xFF;
        }
        // set the bottom bit to 1 to ensure q is odd (better for finding primes)
        qbuf[q_bytes-1] |= 0x01;
        // excess bits are removed 
        qbuf[0] %= q_mask_MSB;
        // set the top (MSB) bit to 1 to ensure that 'q' is relatively large
        qbuf[0] |= (q_mask_MSB >> 1);
    
        mpz_import(q, q_bytes, 1, sizeof(qbuf[0]), 0, 0, qbuf);
        /* next prime */ 
        while(miller_rabin(q, 64) == 0) {mpz_add_ui(q, q, 2);}
    }while(mpz_cmp(p, q) == 0); /* If we have indentical primes (unlikely), try again*/
    free(qbuf);

    /* calculate modulus n = p * q */
    mpz_mul(n, p, q);

    /* calculate totient phi(n) = (p-1)*(q-1) */
    mpz_sub_ui(tmp1, p, 1);
    mpz_sub_ui(tmp2, q, 1);
    mpz_mul(phi, tmp1, tmp2);
    // printf ("totient: "); mpz_out_str(stdout, 10, phi); printf ("\n");

    /* choose public key */
    gcd(tmp1, e, phi);
    while(mpz_cmp_ui(tmp1, 1) != 0) {
        mpz_add_ui(e, e, 2);
        while(miller_rabin(e, 64) == 0) {mpz_add_ui(e, e, 2);}
        gcd(tmp1, e, phi);
    }

    /* compute private key */
    invert(d, e, phi);
    while(mpz_cmp_ui(d, 0) < 0) {
        mpz_add(d, d, phi);
    }

    printf ("0x"); mpz_out_str(stdout, 16, p); printf (" ");
    printf ("0x"); mpz_out_str(stdout, 16, q); printf (" ");
    printf ("0x"); mpz_out_str(stdout, 16, n); printf (" ");
    printf ("0x"); mpz_out_str(stdout, 16, e); printf (" ");
    printf ("0x"); mpz_out_str(stdout, 16, d); printf ("\n");

    mpz_clears(p, q, n, phi, tmp1, tmp2, e, d, NULL);
}

/* EN/DECRYPTION RSA */

void rsa_endecrypt(struct rsa_class *rsa_input) {
    mpz_t encrypted_message;
    mpz_init(encrypted_message);
    
    mpz_powm(encrypted_message, rsa_input->message, rsa_input->exponent, rsa_input->modulus);
    printf ("0x"); mpz_out_str(stdout, 16, encrypted_message); printf ("\n");
    
    mpz_clear(encrypted_message);
}

/* BREAKING RSA */

void pollard_rho(mpz_t factor, const mpz_t n, gmp_randstate_t rnd_state) {
    mpz_t xi, xm, s, tmp_abs, i, i_minus_1, and, n_minus_2;

    mpz_inits(xi, xm, s, tmp_abs, i_minus_1, and, NULL);
    mpz_init_set_ui(i, 0);
    mpz_init_set(n_minus_2, n);

    mpz_sub_ui(n_minus_2, n_minus_2, 2);
    mpz_urandomm(xi, rnd_state, n_minus_2);
    mpz_add_ui(xi, xi, 2);

    mpz_set(xm, xi);
    mpz_set_ui(s, 1);

    while (mpz_cmp_ui(s, 1) == 0) {
        mpz_powm_ui(xi, xi, 2, n);
        mpz_add_ui(xi, xi, 1); 
        mpz_mod(xi, xi, n);

        mpz_sub(tmp_abs, xi, xm);
        mpz_abs(tmp_abs, tmp_abs);
        mpz_gcd(s, tmp_abs, n);

        mpz_sub_ui(i_minus_1, i, 1);
        mpz_and(and, i, i_minus_1);
        if (mpz_cmp_ui(and, 0) == 0 /* i is a power of 2 */) { 
            mpz_set(xm, xi);
        }
        mpz_add_ui(i,i, 1);
    }
    mpz_set(factor, s);
    
    mpz_clear(xi);
    mpz_clear(xm);
    mpz_clear(s);
    mpz_clear(tmp_abs);
}

int trial_division(mpz_t p, mpz_t q, const mpz_t n) {
    unsigned int index = 0;
    mpz_t i, mod;
    mpz_inits(i, mod, NULL);

    for (mpz_set_ui(i, prime_numbers[index++]); index < PRIMES_COUNT; mpz_set_ui(i,prime_numbers[index++])) {
        mpz_mod(mod, n, i);
        if (mpz_cmp_ui(mod, 0) == 0 && mpz_cmp(n, i) != 0) {
            mpz_set(p, i);
            mpz_tdiv_q(q, n, p);
            return 1;
        }
    }
    return 0;
}

int rsa_break(struct rsa_class *rsa_input) {
    /* is modulus prime ? */
    if (miller_rabin(rsa_input->modulus, 64)) {
        fprintf(stderr, "ERROR: Modulus is a prime, has no factors!\n");
        return 1;
    }

    mpz_t p,q, tmp1, tmp2, phi, d, m;
    mpz_inits(p, q, tmp1, tmp2, phi, d, m, NULL);

    gmp_randstate_t rnd_state;
    gmp_randinit_default(rnd_state);
    gmp_randseed_ui(rnd_state, urandom());

    if (!trial_division(p, q, rsa_input->modulus)) {
        pollard_rho(p, rsa_input->modulus, rnd_state);
        while(mpz_cmp(p, rsa_input->modulus) == 0) {
            gmp_randseed_ui(rnd_state, urandom());
            pollard_rho(p, rsa_input->modulus, rnd_state);
        }
        mpz_tdiv_q(q, rsa_input->modulus, p);
    } 

    printf("0x"); mpz_out_str(stdout, 16, p); printf(" ");
    printf("0x"); mpz_out_str(stdout, 16, q); printf(" ");

    /* calculate totient phi(n) = (p-1)*(q-1) */
    mpz_sub_ui(tmp1, p, 1);
    mpz_sub_ui(tmp2, q, 1);
    mpz_mul(phi, tmp1, tmp2);

    /* compute private key */
    mpz_invert(d, rsa_input->exponent, phi);
    while(mpz_cmp_ui(d, 0) < 0) {
        mpz_add(d, d, phi);
    }
    // printf("d: 0x"); mpz_out_str(stdout, 16, d); printf("\n");
    mpz_powm(m, rsa_input->message, d, rsa_input->modulus);
    printf("0x"); mpz_out_str(stdout, 16, m); printf("\n");

    gmp_randclear(rnd_state);
    mpz_clears(p, q, tmp1, tmp2, phi, d, m, NULL);

    return 0;
}