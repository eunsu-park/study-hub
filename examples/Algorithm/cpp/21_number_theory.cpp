/*
 * Number Theory
 * GCD/LCM, Prime, Modular Arithmetic, Combinatorics
 *
 * Algorithms for solving mathematical problems.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace std;

const long long MOD = 1e9 + 7;

// =============================================================================
// 1. GCD / LCM
// =============================================================================

long long gcd(long long a, long long b) {
    while (b) {
        a %= b;
        swap(a, b);
    }
    return a;
}

long long lcm(long long a, long long b) {
    return a / gcd(a, b) * b;
}

// Extended Euclidean Algorithm
// Returns x, y satisfying ax + by = gcd(a, b)
tuple<long long, long long, long long> extendedGcd(long long a, long long b) {
    if (b == 0) {
        return {a, 1, 0};
    }
    auto [g, x1, y1] = extendedGcd(b, a % b);
    return {g, y1, x1 - (a / b) * y1};
}

// =============================================================================
// 2. Primality Testing and Generation
// =============================================================================

bool isPrime(long long n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;

    for (long long i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

// Sieve of Eratosthenes
vector<bool> sieveOfEratosthenes(int n) {
    vector<bool> isPrime(n + 1, true);
    isPrime[0] = isPrime[1] = false;

    for (int i = 2; i * i <= n; i++) {
        if (isPrime[i]) {
            for (int j = i * i; j <= n; j += i) {
                isPrime[j] = false;
            }
        }
    }

    return isPrime;
}

// Prime factorization
vector<pair<long long, int>> factorize(long long n) {
    vector<pair<long long, int>> factors;

    for (long long i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            int cnt = 0;
            while (n % i == 0) {
                n /= i;
                cnt++;
            }
            factors.push_back({i, cnt});
        }
    }

    if (n > 1) {
        factors.push_back({n, 1});
    }

    return factors;
}

// =============================================================================
// 3. Modular Arithmetic
// =============================================================================

// (a + b) % mod
long long addMod(long long a, long long b, long long mod = MOD) {
    return ((a % mod) + (b % mod)) % mod;
}

// (a * b) % mod
long long mulMod(long long a, long long b, long long mod = MOD) {
    return ((a % mod) * (b % mod)) % mod;
}

// (a ^ b) % mod (fast exponentiation)
long long powMod(long long a, long long b, long long mod = MOD) {
    long long result = 1;
    a %= mod;

    while (b > 0) {
        if (b & 1) {
            result = mulMod(result, a, mod);
        }
        a = mulMod(a, a, mod);
        b >>= 1;
    }

    return result;
}

// Modular inverse (Fermat's little theorem, when mod is prime)
long long modInverse(long long a, long long mod = MOD) {
    return powMod(a, mod - 2, mod);
}

// Modular inverse (Extended Euclidean)
long long modInverseExtGcd(long long a, long long mod) {
    auto [g, x, y] = extendedGcd(a, mod);
    if (g != 1) return -1;  // No inverse
    return (x % mod + mod) % mod;
}

// =============================================================================
// 4. Combinatorics
// =============================================================================

class Combination {
private:
    vector<long long> fact;
    vector<long long> invFact;
    long long mod;

public:
    Combination(int n, long long mod = MOD) : mod(mod) {
        fact.resize(n + 1);
        invFact.resize(n + 1);

        fact[0] = 1;
        for (int i = 1; i <= n; i++) {
            fact[i] = mulMod(fact[i-1], i, mod);
        }

        invFact[n] = powMod(fact[n], mod - 2, mod);
        for (int i = n - 1; i >= 0; i--) {
            invFact[i] = mulMod(invFact[i+1], i + 1, mod);
        }
    }

    // nCr
    long long C(int n, int r) {
        if (r < 0 || r > n) return 0;
        return mulMod(fact[n], mulMod(invFact[r], invFact[n-r], mod), mod);
    }

    // nPr
    long long P(int n, int r) {
        if (r < 0 || r > n) return 0;
        return mulMod(fact[n], invFact[n-r], mod);
    }

    // Catalan number
    long long catalan(int n) {
        return mulMod(C(2 * n, n), modInverse(n + 1, mod), mod);
    }
};

// =============================================================================
// 5. Euler's Totient Function
// =============================================================================

long long eulerPhi(long long n) {
    long long result = n;

    for (long long i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            while (n % i == 0) {
                n /= i;
            }
            result -= result / i;
        }
    }

    if (n > 1) {
        result -= result / n;
    }

    return result;
}

// Euler's totient function for 1~n
vector<int> eulerPhiSieve(int n) {
    vector<int> phi(n + 1);
    iota(phi.begin(), phi.end(), 0);

    for (int i = 2; i <= n; i++) {
        if (phi[i] == i) {  // i is prime
            for (int j = i; j <= n; j += i) {
                phi[j] -= phi[j] / i;
            }
        }
    }

    return phi;
}

// =============================================================================
// 6. Chinese Remainder Theorem (CRT)
// =============================================================================

// x = a1 (mod m1), x = a2 (mod m2)
pair<long long, long long> crt(long long a1, long long m1,
                                long long a2, long long m2) {
    auto [g, p, q] = extendedGcd(m1, m2);

    if ((a2 - a1) % g != 0) {
        return {-1, -1};  // No solution
    }

    long long l = m1 / g * m2;  // lcm
    long long x = ((a1 + m1 * (((a2 - a1) / g * p) % (m2 / g))) % l + l) % l;

    return {x, l};
}

// =============================================================================
// 7. Miller-Rabin Primality Test (Large Numbers)
// =============================================================================

long long mulModLarge(long long a, long long b, long long mod) {
    return (__int128)a * b % mod;
}

long long powModLarge(long long a, long long b, long long mod) {
    long long result = 1;
    a %= mod;
    while (b > 0) {
        if (b & 1) result = mulModLarge(result, a, mod);
        a = mulModLarge(a, a, mod);
        b >>= 1;
    }
    return result;
}

bool millerRabin(long long n, long long a) {
    if (n % a == 0) return n == a;

    long long d = n - 1;
    int r = 0;
    while (d % 2 == 0) {
        d /= 2;
        r++;
    }

    long long x = powModLarge(a, d, n);
    if (x == 1 || x == n - 1) return true;

    for (int i = 0; i < r - 1; i++) {
        x = mulModLarge(x, x, n);
        if (x == n - 1) return true;
    }

    return false;
}

bool isPrimeLarge(long long n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;

    vector<long long> witnesses = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (long long a : witnesses) {
        if (n == a) return true;
        if (!millerRabin(n, a)) return false;
    }
    return true;
}

// =============================================================================
// 8. Divisor Functions
// =============================================================================

// List of divisors
vector<long long> getDivisors(long long n) {
    vector<long long> divisors;

    for (long long i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            divisors.push_back(i);
            if (i != n / i) {
                divisors.push_back(n / i);
            }
        }
    }

    sort(divisors.begin(), divisors.end());
    return divisors;
}

// Number of divisors
int countDivisors(long long n) {
    int count = 0;
    for (long long i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            count++;
            if (i != n / i) count++;
        }
    }
    return count;
}

// =============================================================================
// Test
// =============================================================================

int main() {
    cout << "============================================================" << endl;
    cout << "Number Theory Example" << endl;
    cout << "============================================================" << endl;

    // 1. GCD / LCM
    cout << "\n[1] GCD / LCM" << endl;
    cout << "    gcd(48, 18) = " << gcd(48, 18) << endl;
    cout << "    lcm(48, 18) = " << lcm(48, 18) << endl;
    auto [g, x, y] = extendedGcd(48, 18);
    cout << "    48*" << x << " + 18*" << y << " = " << g << endl;

    // 2. Primes
    cout << "\n[2] Primality Testing" << endl;
    cout << "    17 is prime: " << (isPrime(17) ? "yes" : "no") << endl;
    cout << "    100 is prime: " << (isPrime(100) ? "yes" : "no") << endl;

    auto sieve = sieveOfEratosthenes(30);
    cout << "    Primes up to 30: ";
    for (int i = 2; i <= 30; i++) {
        if (sieve[i]) cout << i << " ";
    }
    cout << endl;

    // 3. Prime Factorization
    cout << "\n[3] Prime Factorization" << endl;
    cout << "    360 = ";
    auto factors = factorize(360);
    for (size_t i = 0; i < factors.size(); i++) {
        cout << factors[i].first << "^" << factors[i].second;
        if (i < factors.size() - 1) cout << " x ";
    }
    cout << endl;

    // 4. Modular Arithmetic
    cout << "\n[4] Modular Arithmetic" << endl;
    cout << "    2^10 mod 1000 = " << powMod(2, 10, 1000) << endl;
    cout << "    Inverse of 3 mod 7 = " << modInverse(3, 7) << endl;

    // 5. Combinatorics
    cout << "\n[5] Combinatorics" << endl;
    Combination comb(100);
    cout << "    C(10, 3) = " << comb.C(10, 3) << endl;
    cout << "    P(10, 3) = " << comb.P(10, 3) << endl;
    cout << "    Catalan(5) = " << comb.catalan(5) << endl;

    // 6. Euler's Totient Function
    cout << "\n[6] Euler's Totient Function" << endl;
    cout << "    phi(12) = " << eulerPhi(12) << endl;
    cout << "    phi(36) = " << eulerPhi(36) << endl;

    // 7. CRT
    cout << "\n[7] Chinese Remainder Theorem" << endl;
    auto [result, mod] = crt(2, 3, 3, 5);
    cout << "    x = 2 (mod 3), x = 3 (mod 5)" << endl;
    cout << "    x = " << result << " (mod " << mod << ")" << endl;

    // 8. Divisors
    cout << "\n[8] Divisors" << endl;
    cout << "    Divisors of 36: ";
    for (auto d : getDivisors(36)) {
        cout << d << " ";
    }
    cout << endl;
    cout << "    Number of divisors of 36: " << countDivisors(36) << endl;

    // 9. Complexity Summary
    cout << "\n[9] Complexity Summary" << endl;
    cout << "    | Algorithm         | Time              |" << endl;
    cout << "    |-------------------|-------------------|" << endl;
    cout << "    | GCD (Euclidean)   | O(log min(a,b))   |" << endl;
    cout << "    | Primality test    | O(sqrt(n))        |" << endl;
    cout << "    | Eratosthenes      | O(n log log n)    |" << endl;
    cout << "    | Factorization     | O(sqrt(n))        |" << endl;
    cout << "    | Fast exponent.    | O(log n)          |" << endl;
    cout << "    | Miller-Rabin      | O(k log^2 n)      |" << endl;

    cout << "\n============================================================" << endl;

    return 0;
}
