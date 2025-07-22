# Implementation of Shor's Algorithm to factor N=15
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram
from qiskit.primitives import Sampler  # Use this instead of execute
from math import gcd
from fractions import Fraction
import matplotlib.pyplot as plt

def c_amod15(a, power):
    """Controlled multiplication by a mod 15"""
    if a not in [2, 4, 7, 8, 11, 13]:
        raise ValueError("'a' must be 2, 4, 7, 8, 11 or 13")
    
    U = QuantumCircuit(4)
    
    # Iterate through each bit of power
    for iteration in range(power):
        if a == 2:
            # Controlled SWAP gate
            U.cswap(0, 1, 3)
            U.cswap(0, 2, 1)
            U.cswap(0, 3, 2)
        elif a == 4:
            # Controlled double SWAP gate
            U.cswap(0, 2, 3)
            U.cswap(0, 1, 2)
            U.cswap(0, 0, 1)
        elif a == 7:
            # Controlled x^7 mod 15 operation
            U.cx(0, 1)
            U.cx(0, 2)
            U.cx(0, 3)
            U.ccx(0, 1, 2)
            U.ccx(0, 2, 3)
        elif a == 8:
            # Controlled SWAP gate for a=8
            U.cswap(0, 0, 3)
            U.cswap(0, 1, 2)
        elif a == 11:
            # Controlled x^11 mod 15 operation
            U.cx(0, 0)
            U.cx(0, 1)
            U.ccx(0, 0, 2)
            U.ccx(0, 2, 3)
        elif a == 13:
            # Controlled x^13 mod 15 operation
            U.cx(0, 1)
            U.cx(0, 3)
            U.ccx(0, 1, 0)
            U.ccx(0, 0, 2)
            
    return U

def qft_dagger(n):
    """Quantum Fourier Transform Dagger (Inverse QFT)"""
    qc = QuantumCircuit(n)
    # Apply the inverse QFT
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    
    return qc

def qft(n):
    """Quantum Fourier Transform"""
    qc = QuantumCircuit(n)
    
    for j in range(n):
        qc.h(j)
        for k in range(j+1, n):
            qc.cp(np.pi/float(2**(k-j)), j, k)
    
    return qc

def display_circuit(circuit, title="Quantum Circuit"):
    """
    Display the quantum circuit with proper formatting
    
    Args:
        circuit (QuantumCircuit): The quantum circuit to display
        title (str): Title for the circuit diagram
    """
    # Set figure size based on circuit size
    n_qubits = circuit.num_qubits
    n_gates = len(circuit)
    
    # Calculate reasonable figure dimensions
    fig_width = max(10, n_gates * 0.15)
    fig_height = max(5, n_qubits * 0.5)
    
    plt.figure(figsize=(fig_width, fig_height))
    
    # Draw the circuit
    circuit_drawing = circuit.draw(output='mpl', 
                                  style={'name': 'bw'},
                                  fold=n_gates//20 if n_gates > 40 else 0,
                                  plot_barriers=True,
                                  initial_state=True)
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    print(f"Circuit details:")
    print(f"- Number of qubits: {circuit.num_qubits}")
    print(f"- Number of classical bits: {circuit.num_clbits}")
    print(f"- Circuit depth: {circuit.depth()}")
    print(f"- Number of gates: {n_gates}")

def shors_algorithm(N=15, a=2, visualize=False):
    """
    Implements Shor's algorithm to factor N=15 using a=2 as the random number
    
    Args:
        N (int): Number to factor
        a (int): Random number to use for the period finding
        visualize (bool): Whether to visualize the circuit
    
    Returns:
        tuple: The factors of N
    """
    # Check if a is coprime with N
    if gcd(a, N) != 1:
        print(f"Found factor by luck: {gcd(a, N)}")
        return gcd(a, N), N // gcd(a, N)
    
    # Number of counting qubits needed
    n_count = 8  # We need 8 qubits to store the phase
    
    # Create quantum and classical registers
    q_count = QuantumRegister(n_count, 'count')
    q_aux = QuantumRegister(4, 'aux')
    c_out = ClassicalRegister(n_count, 'c')
    
    # Create quantum circuit
    qc = QuantumCircuit(q_count, q_aux, c_out)
    
    # Initialize counting qubits in superposition
    for q in range(n_count):
        qc.h(q)
    
    # Initialize aux register to |1>
    qc.x(q_aux[0])
    
    # Apply controlled U operations
    for q in range(n_count):
        # Apply U^(2^q) controlled by qth counting qubit
        qc.append(c_amod15(a, 2**q).control(), [q_count[q]] + [q_aux[i] for i in range(4)])
    
    # Apply inverse QFT on counting qubits
    qc.append(qft_dagger(n_count), q_count)
    
    # Measure counting qubits
    qc.measure(q_count, c_out)
    
    # Display the circuit if requested
    if visualize:
        display_circuit(qc, f"Shor's Algorithm Circuit for Factoring {N} with a={a}")
    
    # Simulate the circuit using the Sampler primitive
    sampler = Sampler()
    job = sampler.run(qc, shots=1024)
    result = job.result()
    
    # Get the quasi-probability distribution
    quasi_dist = result.quasi_dists[0]
    
    # Convert to a more standard counts format
    counts = {}
    for bitstring, probability in quasi_dist.items():
        # Convert integer outcome to binary string of appropriate length
        binary = format(bitstring, f'0{n_count}b')
        # Store only non-zero probabilities with actual shot counts
        if probability > 0:
            counts[binary] = int(probability * 1024)  # Scale by number of shots
    
    print("Circuit executed")
    print("Measurement results:", counts)
    
    # Visualize the measurement histogram if requested
    if visualize:
        plot_histogram(counts, title=f"Measurement Results for N={N}, a={a}")
        plt.show()
    
    # Process the results
    factors = process_results(a, N, n_count, counts)
    return factors

def process_results(a, N, n_count, counts):
    """Process the measurement results to find factors"""
    factors_found = []
    
    # Sort by descending count frequency
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    print("\nProcessing results:")
    for output, count in sorted_counts:
        # Convert binary to decimal
        decimal = int(output, 2)
        
        # Skip if decimal is 0
        if decimal == 0:
            continue
        
        # Calculate phase from decimal value
        phase = decimal / (2**n_count)
        
        # Find closest fraction with continued fractions
        frac = Fraction(phase).limit_denominator(N)
        r = frac.denominator
        
        print(f"Decimal: {decimal}, Phase: {phase}, Fraction: {frac}, r: {r}")
        
        # Ensure r is even for the algorithm to work
        if r % 2 == 1 and r < N:
            r *= 2
            print(f"  r is odd, doubling to: {r}")
        
        # Compute the guessed factors
        if r % 2 == 0 and r < N:
            guessed_factor1 = gcd(a**(r//2) - 1, N)
            guessed_factor2 = gcd(a**(r//2) + 1, N)
            
            if guessed_factor1 not in [1, N] and guessed_factor2 not in [1, N]:
                print(f"  Success! Factors: {guessed_factor1} and {guessed_factor2}")
                if [guessed_factor1, guessed_factor2] not in factors_found and [guessed_factor2, guessed_factor1] not in factors_found:
                    factors_found.append([guessed_factor1, guessed_factor2])
            else:
                print(f"  Failed to find non-trivial factors with this r value.")
    
    if not factors_found:
        print("No factors found. Try another value of a or increase the number of shots.")
        return None
    else:
        # Return the first valid factorization found
        return factors_found[0]

# Run Shor's algorithm to factor N=15
print("Running Shor's algorithm to factor N=15...")
factors = shors_algorithm(15, 7, visualize=True)
if factors:
    print(f"\nFactors of 15: {factors[0]} and {factors[1]}")
else:
    print("\nNo factors found. Try running the algorithm again.")