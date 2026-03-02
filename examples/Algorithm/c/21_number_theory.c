/*
 * Number Theory
 * GCD/LCM, Primes, Modular Arithmetic, Combinatorics
 *
 * Foundations of mathematical algorithms.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#define MOD 1000000007
#define MAX_N 100001

/* =============================================================================
 * 1. GCD / LCM
 * ============================================================================= */

long long gcd(long long a, long long b) {
    while (b) {
        long long temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

long long gcd_recursive(long long a, long long b) {
    if (b == 0) return a;
    return gcd_recursive(b, a % b);
}

long long lcm(long long a, long long b) {
    return a / gcd(a, b) * b;
}

/* Extended Euclidean Algorithm: ax + by = gcd(a, b) */
long long extended_gcd(long long a, long long b, long long* x, long long* y) {
    if (b == 0) {
        *x = 1;
        *y = 0;
        return a;
    }
    long long x1, y1;
    long long g = extended_gcd(b, a % b, &x1, &y1);
    *x = y1;
    *y = x1 - (a / b) * y1;
    return g;
}

/* =============================================================================
 * 2. Primality Test and Sieves
 * ============================================================================= */

bool is_prime(long long n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (long long i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

/* Sieve of Eratosthenes */
bool* sieve_of_eratosthenes(int n) {
    bool* is_prime_arr = malloc((n + 1) * sizeof(bool));
    memset(is_prime_arr, true, n + 1);
    is_prime_arr[0] = is_prime_arr[1] = false;

    for (int i = 2; i * i <= n; i++) {
        if (is_prime_arr[i]) {
            for (int j = i * i; j <= n; j += i) {
                is_prime_arr[j] = false;
            }
        }
    }
    return is_prime_arr;
}

/* Linear Sieve (generates prime list) */
int* linear_sieve(int n, int* prime_count) {
    int* spf = calloc(n + 1, sizeof(int));  /* smallest prime factor */
    int* primes = malloc((n + 1) * sizeof(int));
    *prime_count = 0;

    for (int i = 2; i <= n; i++) {
        if (spf[i] == 0) {
            spf[i] = i;
            primes[(*prime_count)++] = i;
        }
        for (int j = 0; j < *prime_count && primes[j] <= spf[i] && i * primes[j] <= n; j++) {
            spf[i * primes[j]] = primes[j];
        }
    }

    free(spf);
    return primes;
}

/* Prime Factorization */
typedef struct {
    int prime;
    int count;
} PrimeFactor;

int factorize(long long n, PrimeFactor factors[]) {
    int count = 0;
    for (long long d = 2; d * d <= n; d++) {
        if (n % d == 0) {
            factors[count].prime = d;
            factors[count].count = 0;
            while (n % d == 0) {
                factors[count].count++;
                n /= d;
            }
            count++;
        }
    }
    if (n > 1) {
        factors[count].prime = n;
        factors[count].count = 1;
        count++;
    }
    return count;
}

/* =============================================================================
 * 3. Modular Arithmetic
 * ============================================================================= */

long long mod_add(long long a, long long b, long long mod) {
    return ((a % mod) + (b % mod)) % mod;
}

long long mod_sub(long long a, long long b, long long mod) {
    return ((a % mod) - (b % mod) + mod) % mod;
}

long long mod_mul(long long a, long long b, long long mod) {
    return ((a % mod) * (b % mod)) % mod;
}

/* Fast Exponentiation */
long long mod_pow(long long base, long long exp, long long mod) {
    long long result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) {
            result = mod_mul(result, base, mod);
        }
        base = mod_mul(base, base, mod);
        exp >>= 1;
    }
    return result;
}

/* Modular Inverse (Fermat's Little Theorem, mod must be prime) */
long long mod_inverse(long long a, long long mod) {
    return mod_pow(a, mod - 2, mod);
}

/* Modular Inverse (Extended Euclidean) */
long long mod_inverse_ext(long long a, long long mod) {
    long long x, y;
    long long g = extended_gcd(a, mod, &x, &y);
    if (g != 1) return -1;  /* No inverse */
    return (x % mod + mod) % mod;
}

/* Modular Division */
long long mod_div(long long a, long long b, long long mod) {
    return mod_mul(a, mod_inverse(b, mod), mod);
}

/* =============================================================================
 * 4. Combinatorics
 * ============================================================================= */

long long factorial[MAX_N];
long long inv_factorial[MAX_N];

void precompute_factorials(int n, long long mod) {
    factorial[0] = 1;
    for (int i = 1; i <= n; i++) {
        factorial[i] = mod_mul(factorial[i - 1], i, mod);
    }
    inv_factorial[n] = mod_inverse(factorial[n], mod);
    for (int i = n - 1; i >= 0; i--) {
        inv_factorial[i] = mod_mul(inv_factorial[i + 1], i + 1, mod);
    }
}

/* nCr (mod p) */
long long nCr(int n, int r, long long mod) {
    if (r < 0 || r > n) return 0;
    return mod_mul(factorial[n], mod_mul(inv_factorial[r], inv_factorial[n - r], mod), mod);
}

/* nPr (mod p) */
long long nPr(int n, int r, long long mod) {
    if (r < 0 || r > n) return 0;
    return mod_mul(factorial[n], inv_factorial[n - r], mod);
}

/* Pascal's Triangle (for small n) */
long long** pascal_triangle(int n) {
    long long** C = malloc((n + 1) * sizeof(long long*));
    for (int i = 0; i <= n; i++) {
        C[i] = calloc(n + 1, sizeof(long long));
        C[i][0] = 1;
        for (int j = 1; j <= i; j++) {
            C[i][j] = (C[i - 1][j - 1] + C[i - 1][j]) % MOD;
        }
    }
    return C;
}

/* =============================================================================
 * 5. Euler's Totient Function
 * ============================================================================= */

long long euler_phi(long long n) {
    long long result = n;
    for (long long p = 2; p * p <= n; p++) {
        if (n % p == 0) {
            while (n % p == 0) n /= p;
            result -= result / p;
        }
    }
    if (n > 1) result -= result / n;
    return result;
}

/* Euler Totient Sieve */
int* euler_phi_sieve(int n) {
    int* phi = malloc((n + 1) * sizeof(int));
    for (int i = 0; i <= n; i++) phi[i] = i;

    for (int i = 2; i <= n; i++) {
        if (phi[i] == i) {  /* i is prime */
            for (int j = i; j <= n; j += i) {
                phi[j] -= phi[j] / i;
            }
        }
    }
    return phi;
}

/* =============================================================================
 * 6. Chinese Remainder Theorem (CRT)
 * ============================================================================= */

/* x = a1 (mod m1), x = a2 (mod m2) */
long long crt_two(long long a1, long long m1, long long a2, long long m2) {
    long long x, y;
    long long g = extended_gcd(m1, m2, &x, &y);

    if ((a2 - a1) % g != 0) return -1;  /* No solution */

    long long lcm_val = m1 / g * m2;
    long long result = a1 + m1 * ((a2 - a1) / g % (m2 / g) * x % (m2 / g));
    return ((result % lcm_val) + lcm_val) % lcm_val;
}

/* =============================================================================
 * 7. Divisors
 * ============================================================================= */

int count_divisors(long long n) {
    int count = 0;
    for (long long i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            count++;
            if (i != n / i) count++;
        }
    }
    return count;
}

long long sum_divisors(long long n) {
    long long sum = 0;
    for (long long i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            sum += i;
            if (i != n / i) sum += n / i;
        }
    }
    return sum;
}

int* get_divisors(long long n, int* count) {
    int* divisors = malloc(10000 * sizeof(int));
    *count = 0;
    for (long long i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            divisors[(*count)++] = i;
            if (i != n / i) divisors[(*count)++] = n / i;
        }
    }
    return divisors;
}

/* =============================================================================
 * Test
 * ============================================================================= */

int compare_int(const void* a, const void* b) {
    return *(int*)a - *(int*)b;
}

int main(void) {
    printf("============================================================\n");
    printf("Number Theory Examples\n");
    printf("============================================================\n");

    /* 1. GCD / LCM */
    printf("\n[1] GCD / LCM\n");
    printf("    gcd(48, 18) = %lld\n", gcd(48, 18));
    printf("    lcm(12, 18) = %lld\n", lcm(12, 18));

    long long x, y;
    long long g = extended_gcd(35, 15, &x, &y);
    printf("    35*%lld + 15*%lld = %lld (gcd)\n", x, y, g);

    /* 2. Primes */
    printf("\n[2] Primality Test and Sieves\n");
    printf("    is_prime(17) = %s\n", is_prime(17) ? "true" : "false");
    printf("    is_prime(18) = %s\n", is_prime(18) ? "true" : "false");

    int prime_count;
    int* primes = linear_sieve(50, &prime_count);
    printf("    Primes up to 50 (%d total): ", prime_count);
    for (int i = 0; i < prime_count; i++) printf("%d ", primes[i]);
    printf("\n");
    free(primes);

    /* 3. Prime Factorization */
    printf("\n[3] Prime Factorization\n");
    PrimeFactor factors[20];
    int factor_count = factorize(360, factors);
    printf("    360 = ");
    for (int i = 0; i < factor_count; i++) {
        printf("%d^%d", factors[i].prime, factors[i].count);
        if (i < factor_count - 1) printf(" * ");
    }
    printf("\n");

    /* 4. Modular Arithmetic */
    printf("\n[4] Modular Arithmetic\n");
    printf("    2^10 mod 1000 = %lld\n", mod_pow(2, 10, 1000));
    printf("    3^(-1) mod 7 = %lld\n", mod_inverse(3, 7));
    printf("    Verify: 3 * %lld mod 7 = %lld\n", mod_inverse(3, 7),
           mod_mul(3, mod_inverse(3, 7), 7));

    /* 5. Combinatorics */
    printf("\n[5] Combinatorics\n");
    precompute_factorials(1000, MOD);
    printf("    10! = %lld\n", factorial[10]);
    printf("    C(10, 3) = %lld\n", nCr(10, 3, MOD));
    printf("    P(10, 3) = %lld\n", nPr(10, 3, MOD));

    /* 6. Euler's Totient */
    printf("\n[6] Euler's Totient Function\n");
    printf("    phi(12) = %lld\n", euler_phi(12));
    printf("    phi(13) = %lld (prime)\n", euler_phi(13));

    /* 7. Divisors */
    printf("\n[7] Divisors\n");
    printf("    Divisor count of 28: %d\n", count_divisors(28));
    printf("    Divisor sum of 28: %lld\n", sum_divisors(28));
    int div_count;
    int* divisors = get_divisors(28, &div_count);
    qsort(divisors, div_count, sizeof(int), compare_int);
    printf("    Divisors of 28: ");
    for (int i = 0; i < div_count; i++) printf("%d ", divisors[i]);
    printf("\n");
    free(divisors);

    /* 8. Complexity */
    printf("\n[8] Complexity\n");
    printf("    | Algorithm        | Time          |\n");
    printf("    |------------------|---------------|\n");
    printf("    | GCD (Euclidean)  | O(log min)    |\n");
    printf("    | Primality test   | O(sqrt(n))    |\n");
    printf("    | Eratosthenes     | O(n log log n)|\n");
    printf("    | Factorization    | O(sqrt(n))    |\n");
    printf("    | Exponentiation   | O(log exp)    |\n");

    printf("\n============================================================\n");

    return 0;
}
