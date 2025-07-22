from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
import numpy as np
import math
from fractions import Fraction
from math import gcd

def c_amod15(a, power):
    """
    Controlled multiplication by a mod 15
    
    Args:
        a: the factor to multiply by
        power: the number of times to multiply
    
    Returns:
        qc: quantum circuit implementing the controlled multiplication
    """
    if a not in [2, 4, 7, 8, 11, 13]:
        raise ValueError("'a' must be 2, 4, 7, 8, 11 or 13")
    
    U = QuantumCircuit(4)
    
    for _ in range(power):
        if a == 2:
            # Multiply by 2 mod 15
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
        elif a == 4:
            # Multiply by 4 mod 15
            U.swap(0, 2)
            U.swap(1, 3)
        elif a == 7:
            # Multiply by 7 mod 15
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
            # Then apply inverse of 8 (which is 2)
            U.x(0)
            U.x(1)
            U.x(2)
            U.x(3)
        elif a == 8:
            # Multiply by 8 mod 15
            U.swap(0, 3)
            U.swap(1, 2)
        elif a == 11:
            # Multiply by 11 mod 15
            U.swap(0, 3)
            U.swap(1, 2)
            # Then apply inverse of 4
            U.x(0)
            U.x(1)
            U.x(2)
            U.x(3)
        elif a == 13:
            # Multiply by 13 mod 15
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
            # Then apply inverse of 2
            U.x(0)
            U.x(1)
            U.x(2)
            U.x(3)
    
    return U

def qft_dagger(n):
    """
    Quantum Fourier Transform Inverse
    
    Args:
        n: number of qubits
    
    Returns:
        qc: quantum circuit implementing the inverse QFT
    """
    qc = QuantumCircuit(n)
    
    # Apply the inverse QFT
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    
    return qc

def create_shor_circuit(n, a):
    """
    Create Shor's algorithm quantum circuit for factoring N=15
    
    Args:
        n: number of counting qubits
        a: base of the modular exponentiation
    
    Returns:
        qc: quantum circuit implementing Shor's algorithm
    """
    # Create quantum registers
    up_reg = QuantumRegister(n, name='up')  # Counting register
    down_reg = QuantumRegister(4, name='down')  # Work register (for mod 15)
    up_classic = ClassicalRegister(n, name='m')  # Classical register for measurement
    
    # Create quantum circuit
    qc = QuantumCircuit(up_reg, down_reg, up_classic)
    
    # Initialize down register to |1⟩
    qc.x(down_reg[0])
    
    # Apply Hadamard gates to counting qubits
    for i in range(n):
        qc.h(up_reg[i])
    
    # Apply controlled-U operations
    for i in range(n):
        # For qubit i, apply U^(2^i)
        power = 2**i
        qc.append(c_amod15(a, power).control(), [up_reg[i]] + list(down_reg))
    
    # Apply inverse QFT to counting register
    qc.append(qft_dagger(n), up_reg)
    
    # Measure counting register
    qc.measure(up_reg, up_classic)
    
    return qc

def get_factors(N, a, phase, limit=10):
    """
    Get factors from the phase value
    
    Args:
        N: number to factor
        a: base of the modular exponentiation
        phase: measured phase from Shor's algorithm
        limit: limit for the denominator in continued fraction expansion
    
    Returns:
        tuple: possible factors of N
    """
    fraction = Fraction(phase).limit_denominator(limit)
    r = fraction.denominator
    
    print(f"Estimated period: r = {r}")
    
    # Check if r is even
    if r % 2 != 0:
        print("Period is odd, re-run the algorithm.")
        return None
    
    # Calculate potential factors
    guesses = [gcd(a**(r//2) - 1, N), gcd(a**(r//2) + 1, N)]
    
    for guess in guesses:
        if guess not in [1, N]:
            print(f"Found a non-trivial factor: {guess}")
            return guess, N // guess
    
    print("Failed to find a non-trivial factor.")
    return None

def simulate_shor_circuit(qc):
    """
    Simulate Shor's circuit and return the measured phase
    
    Args:
        qc: quantum circuit to simulate
    
    Returns:
        float: measured phase
    """
    # Simulate the circuit
    simulator = AerSimulator()
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit, shots=1).result()
    counts = result.get_counts()
    
    # Get the measured value
    measured_value = int(list(counts.keys())[0], 2)
    phase = measured_value / (2**len(qc.clbits))
    
    return phase

def shor_algorithm(N):
    """
    Run Shor's algorithm to factor N
    
    Args:
        N: number to factor
    
    Returns:
        tuple: factors of N
    """
    print(f"\n{'='*50}")
    print(f"FACTORING {N} USING SHOR'S ALGORITHM")
    print(f"{'='*50}")
    
    # Check if N is even
    if N % 2 == 0:
        return 2, N // 2
    
    if N == 15:
        # For N=15, we can use the specific implementation from the tutorial
        print("Using specific implementation for N=15")
        
        # Choose random base for modular exponentiation
        a = np.random.choice([2, 4, 7, 8, 11, 13])
        print(f"Chosen base a = {a}")
        
        # Create circuit with 8 counting qubits
        qc = create_shor_circuit(8, a)
        
        # Simulate circuit
        phase = simulate_shor_circuit(qc)
        print(f"Measured phase: {phase}")
        
        # Get factors from phase
        factors = get_factors(N, a, phase)
        
        if factors:
            return factors
        
    # For other values, use semi-classical approach
    return semiClassicalShor(N)

def semiClassicalShor(N):
    """
    Implement a semi-classical version of Shor's algorithm
    
    Args:
        N (int): Number to factor
    
    Returns:
        tuple: Two factors of N
    """
    print("Using semi-classical approach to factor", N)
    
    # Try several random bases
    for _ in range(5):
        # Choose random base a where 1 < a < N
        a = np.random.randint(2, N)
        
        # Check if a and N are coprime
        if gcd(a, N) != 1:
            p = gcd(a, N)
            q = N // p
            print(f"Found factors through GCD: {p} × {q}")
            return p, q
        
        print(f"Trying base a = {a}")
        
        # Find period r such that a^r mod N = 1
        r = find_period(a, N)
        print(f"Found period r = {r}")
        
        # Check if r is even and a^(r/2) mod N ≠ -1
        if r % 2 == 0 and pow(a, r // 2, N) != N - 1:
            # Calculate potential factors
            pot_factor1 = gcd(pow(a, r // 2) - 1, N)
            pot_factor2 = gcd(pow(a, r // 2) + 1, N)
            
            if 1 < pot_factor1 < N:
                p = pot_factor1
                q = N // p
                print(f"Found factors through period finding: {p} × {q}")
                return p, q
                
            if 1 < pot_factor2 < N:
                p = pot_factor2
                q = N // p
                print(f"Found factors through period finding: {p} × {q}")
                return p, q
    
    # If we failed with the semi-classical method, use trial division
    print("Semi-classical approach failed, using trial division")
    return trial_division(N)

def find_period(a, N):
    """
    Find the period r where a^r mod N = 1 classically
    
    Args:
        a (int): Base
        N (int): Modulus
    
    Returns:
        int: Period r
    """
    for r in range(1, N):
        if pow(a, r, N) == 1:
            return r
    return 0

def trial_division(N):
    """
    Simple trial division to find factors
    
    Args:
        N (int): Number to factor
    
    Returns:
        tuple: Two factors of N
    """
    for i in range(2, int(math.sqrt(N)) + 1):
        if N % i == 0:
            return i, N // i
    return 1, N  # Prime or factorization failed

def verify_factors(factors, N):
    """
    Verify that the factors multiply to give N
    
    Args:
        factors (tuple): The factors to verify
        N (int): Original number
    
    Returns:
        bool: True if the factors are correct
    """
    p, q = factors
    product = p * q
    
    if product == N:
        print(f"VERIFICATION SUCCESSFUL: {p} × {q} = {N}")
        return True
    else:
        print(f"VERIFICATION FAILED: {p} × {q} = {product}, not {N}")
        return False

def find_prime_factorization(N):
    """
    Find the complete prime factorization of N
    
    Args:
        N (int): Number to factorize
    
    Returns:
        list: List of prime factors
    """
    factors = []
    n = N
    
    # Check for factor 2
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    
    # Check for odd factors
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            factors.append(i)
            n //= i
    
    # If n is a prime number greater than 2
    if n > 2:
        factors.append(n)
        
    return factors

# Main execution
if __name__ == "__main__":
    # Numbers to factor
    numbers = [15, 21]
    
    for N in numbers:
        # Use Shor's algorithm to find factors
        factors = shor_algorithm(N)
        
        # Verify the factorization
        verify_factors(factors, N)
        
        # Find and print prime factorization
        prime_factors = find_prime_factorization(N)
        print(f"Prime factorization of {N}: {' × '.join(map(str, prime_factors))}")
        print(f"{'='*50}\n")