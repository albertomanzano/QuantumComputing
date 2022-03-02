import qat.lang.AQASM as qlm
from qat.lang.AQASM import QRoutine, RY, CNOT, build_gate
import numpy as np
from utils import mask, fwht, test_bins, left_conditional_probability

"""
This module contains all the functions in order to load data into the quantum state.
There are two implementations for the loading of a function: one based on brute force and one based on multiplexors.
The implementation of the multiplexors is a non-recursive version of:

    V.V. Shende, S.S. Bullock, and I.L. Markov.
    Synthesis of quantum-logic circuits.
    IEEE Transactions on Computer-Aided Design of Integrated Circuits
    and Systems, 25(6):1000–1010, Jun 2006
    arXiv:quant-ph/0406176v5
MyQLM version:

"""

# Loading uniform distribution
@qlm.build_gate("UD", [int], arity=lambda x: x)
def uniform_distribution(number_qubits: int):
    """
    Function to load a uniform distribution in a quantum circuit.
    Parameters
    ----------
    number_qubits : int
        Arity of the output gate.
    """
    routine = qlm.QRoutine()
    quantum_register = routine.new_wires(number_qubits)
    for i in range(number_qubits):
        routine.apply(qlm.H,quantum_register[i])
    return routine

@qlm.build_gate("LP", [int,int,float], arity=lambda x,y,z: x)
def load_angle(number_qubits: int, index: int, angle: float):
    """
    Auxiliary function that transforms the state |0>|index> into
    cos(angle)|0>|index>+sin(angle)|1>|index>.
    Parameters
    ----------
    number_qubits : int
        Number of qubits for the control register. The arity of the gate is number_qubits+1.
    index : int
        Index of the state that we control.
    angle : float
        Angle that we load.
    """
    
    routine = qlm.QRoutine()
    quantum_register = routine.new_wires(number_qubits)

    routine.apply(mask(number_qubits-1,index),quantum_register[:number_qubits-1])
    routine.apply(qlm.RY(angle).ctrl(number_qubits-1),\
                  quantum_register[:number_qubits-1],quantum_register[number_qubits-1])
    routine.apply(mask(number_qubits-1,index),quantum_register[:number_qubits-1])

    return routine

def load_angles_brute_force(angles: np.array):
    """
    Creates an Abstract gate using multicontrolled rotations that transforms the state:
    |0>|0>+ |0>|1>+ |0>|2>+...+ |0>|len(angle)-1>,
    into:
    cos(angle)|0>|0>+cos(angle)|0>|1>+cos(angle)|0>|2>+...+cos(angle)|0>|len(angle)-1>
    +sin(angle)|0>|0>+sin(angle)|0>|1>+sin(angle)|0>|2>+...+sin(angle)|0>|len(angle)-1>.
    Parameters
    ----------
    angles : numpy array
        Angles to load in the circuit. The arity of the gate is int(np.log2(len(angle)))+1.
    """
    number_qubits = int(np.log2(angles.size))+1                                             
    routine = qlm.QRoutine()
    quantum_register = routine.new_wires(number_qubits)
    for i in range(angles.size):
        routine.apply(load_angle(number_qubits,i,angles[i]),quantum_register)
    return routine

def multiplexor_RY(angles: np.array,ordering: str="sequency"):
    """
    Creates an Abstract gate using Quantum Multiplexors that transforms the state:
    |0>|0>+ |0>|1>+ |0>|2>+...+ |0>|len(angle)-1>,
    into:
    cos(angle)|0>|0>+cos(angle)|0>|1>+cos(angle)|0>|2>+...+cos(angle)|0>|len(angle)-1>
    +sin(angle)|0>|0>+sin(angle)|0>|1>+sin(angle)|0>|2>+...+sin(angle)|0>|len(angle)-1>.
    Parameters
    ----------
    angles : numpy array
        Angles to load in the circuit. The arity of the gate is int(np.log2(len(angle)))+1.
    """
    number_qubits = int(np.log2(angles.size))
    angles = fwht(angles,ordering =ordering)
    angles = angles/2**number_qubits
    routine = qlm.QRoutine()
    quantum_register = routine.new_wires(number_qubits+1)
    control = np.zeros(2**number_qubits,dtype = int)
    for i in range(number_qubits):
        for j in range(2**i-1,2**number_qubits,2**i):
            control[j] = number_qubits-i-1
    for i in range(2**number_qubits):
        routine.apply(qlm.RY(angles[i]),quantum_register[number_qubits])
        routine.apply(qlm.CNOT,quantum_register[control[i]],quantum_register[number_qubits])
    return routine

def load_angles(angles: np.array,method: str="multiplexor"):
    """
    Auxiliary function that transforms the state:
    |0>|0>+ |0>|1>+ |0>|2>+...+ |0>|len(angle)-1>,
    into:
    cos(angle)|0>|0>+cos(angle)|0>|1>+cos(angle)|0>|2>+...+cos(angle)|0>|len(angle)-1>
    +sin(angle)|0>|0>+sin(angle)|0>|1>+sin(angle)|0>|2>+...+sin(angle)|0>|len(angle)-1>.
    It serves as an interface for the two methods for loading the angles.
    Parameters
    ----------
    angles : numpy array
        Angles to load in the circuit. The arity of the gate is int(np.log2(len(angle)))+1.
    method : string
        Method used in the loading. Default method.
    """
    number_qubits = int(np.log2(angles.size))+1
    if (np.max(angles)>np.pi):
        print("ERROR: function f not properly normalised")
        return
    if (angles.size != 2**(number_qubits-1)):
        print("ERROR: size of function f is not a factor of 2")
    if (method=="brute_force"):
        routine = load_angles_brute_force(angles)
    else:
        routine = multiplexor_RY(angles)
    return routine

def load_probability(probability_array: np.array):
    """
    Creates an Abstract gate for loading the square roots of a discretized
    Probability Distribution using Quantum Multiplexors.

    Parameters
    ----------
    probability_array : numpy array
        Numpy array with the discretized probability to load. The arity of
        of the gate is int(np.log2(len(probability_array))).

    Raises
    ----------
    AssertionError
        if len(probability_array) != 2^n
    """
    number_qubits = int(np.log2(probability_array.size))
    routine = QRoutine()
    register = routine.new_wires(number_qubits)
    # Now go iteratively trough each qubit computing the
    #probabilities and adding the corresponding multiplexor
    for m in range(number_qubits):
        print(m)
        #Calculates Conditional Probability
        conditional_probability = left_conditional_probability(m, probability_array)
        #Rotation angles: length: 2^(i-1)-1 and i the number of
        #qbits of the step
        thetas = 2.0*(np.arccos(np.sqrt(conditional_probability)))
        if m == 0:
            # In the first iteration it is only needed a RY gate
            routine.apply(RY(thetas[0]), register[number_qubits-1])
        else:
            # In the following iterations we have to apply
            # multiplexors controlled by m qubits
            # We call a function to construct the multiplexor,
            # whose action is a block diagonal matrix of Ry gates
            # with angles theta
            routine.apply(multiplexor_RY(thetas),\
                          register[number_qubits-m:number_qubits],register[number_qubits-m-1])
    return routine

