"""
This module contains all the functions in order to load data into the
quantum state using quantum multiplexors.
This module is based in the papper:

    V.V. Shende, S.S. Bullock, and I.L. Markov.
    Synthesis of quantum-logic circuits.
    IEEE Transactions on Computer-Aided Design of Integrated Circuits
    and Systems, 25(6):1000–1010, Jun 2006
    arXiv:quant-ph/0406176v5

Authors: Juan Santos Suárez & Gonzalo Ferro Costas

MyQLM version:

"""

import numpy as np
from qat.lang.AQASM import QRoutine, RY, CNOT, build_gate
from AuxiliarFunctions import test_bins, left_conditional_probability


def multiplexor_ry_m_recurs(qprog, qbits, thetas, r_controls, i_target, sig=1.0):
    """
    Auxiliary function to create the recursive part of a multiplexor
    that applies an RY gate
    Parameters
    ----------

    qprog : Quantum QLM Program
        Quantum Program in which we want to apply the gates
    qbits : int
        Number of qubits of the quantum program
    thetas : np.ndarray
        numpy array containing the set of angles that we want to apply
    r_controls : int
        number of remaining controls
    i_target : int
        index of the target qubits
    sig : float
        accounts for wether our multiplexor is being decomposed with its
        lateral CNOT at the right or at the left, even if that CNOT is
        not present because it cancelled out
        (its values can only be +1. and -1.)
    """
    assert isinstance(r_controls, int), 'm must be an integer'
    assert isinstance(i_target, int), 'j must be an integer'
    assert sig == 1. or sig == -1., 'sig can only be -1. or 1.'
    if  r_controls > 1:
        # If there is more that one control, the multiplexor shall be
        # decomposed. It can be checked that the right way to
        # decompose it taking into account the simplifications is as

        #left angles
        x_l = 0.5*np.array(
            [thetas[i]+sig*thetas[i+len(thetas)//2] for i in range(len(thetas)//2)]
        )

        #right angles
        x_r = 0.5*np.array(
            [thetas[i]-sig*thetas[i+len(thetas)//2] for i in range(len(thetas)//2)]
        )
        multiplexor_ry_m_recurs(qprog, qbits, x_l, r_controls-1, i_target, 1.)
        qprog.apply(CNOT, qbits[i_target-r_controls], qbits[i_target])
        multiplexor_ry_m_recurs(qprog, qbits, x_r, r_controls-1, i_target, -1.)
        # Just for clarification, if we hadn't already simplify the
        # CNOTs, the code should have been
        # if sign == -1.:
        #   multiplexor_ry_m_recurs(qprog, qbits, x_l, r_controls-1, i_target, -1.)
        # qprog.apply(CNOT, qbits[i_target-r_controls], qbits[j])
        # multiplexor_ry_m_recurs(qprog, qbits, x_r, r_controls-1, i_target, -1.)
        # qprog.apply(CNOT, qbits[i_target-r_controls], qbits[i_target])
        # if sign == 1.:
        #   multiplexor_ry_m_recurs(qprog, qbits, x_l, r_controls-1, i_target, 1.)
    else:
        # If there is only one control just apply the Ry gates
        theta_positive = (thetas[0]+sig*thetas[1])/2.0
        theta_negative = (thetas[0]-sig*thetas[1])/2.0
        qprog.apply(RY(theta_positive), qbits[i_target])
        qprog.apply(CNOT, qbits[i_target-1], qbits[i_target])
        qprog.apply(RY(theta_negative), qbits[i_target])

def multiplexor_ry_m(qprog, qbits, thetas, r_controls, i_target):
    """
    Create a multiplexor that applies an RY gate on a qubit controlled
    by the former m qubits. It will have its lateral cnot on the right.
    Given a 2^n vector of thetas this function creates a controlled
    Y rotation of each theta. The rotation is controlled by the basis
    state of a 2^n quantum system.
    If we had a n qbit system and a
        - thetas = [thetas_0, thetas_1, ..., thetas_2^n-1]
    then the function applies
        - RY(thetas_0) controlled by state |0>_{n}
        - RY(thetas_1) controlled by state |1>_{n}
        - RY(thetas_2) controlled by state |2>_{n}
        - ...
        - RY(thetas_2^n-1) controlled by state |2^n-1>_{n}
    On the quantum system.
    Parameters
    ----------

    qprog : Quantum QLM Program
        Quantum Program in which we want to apply the gates
    qbits : int
        Number of qubits of the quantum program
    thetas : np.ndarray
        numpy array containing the set of angles that we want to apply
    r_controls: int
        number of remaining controls
    i_target: int
        index of the target qubits
    """
    multiplexor_ry_m_recurs(qprog, qbits, thetas, r_controls, i_target)
    qprog.apply(CNOT, qbits[i_target-r_controls], qbits[i_target])

def load_p_gate(probability_array):
    """
    Creates a customized AbstractGate for loading a discretized
    Probability using Quantum Multiplexors.

    Parameters
    ----------

    probability_array : numpy array
        Numpy array with the discretized probability to load. The number
        of qbits will be log2(len(probability_array)).

    Raises
    ----------
    AssertionError
        if len(probability_array) != 2^n

    Returns
    ----------

    P_Gate :  AbstractGate
        Customized Abstract Gate for Loading Probability array using
        Quantum Multiplexors
    """

    nqbits = test_bins(probability_array, text='Function')

    @build_gate("P_Gate", [], arity=nqbits)
    def p_gate_qm():
        """
        Function generator for the AbstractGate that allows the loading
        of a discretized Probability in a Quantum State using
        Quantum Multiplexors.
        Returns
        ----------

        Qrout : Quantum Routine
            Quantum Routine for loading Probability using Quantum
            Multiplexors
        """
        qrout = QRoutine()
        reg = qrout.new_wires(nqbits)
        # Now go iteratively trough each qubit computing the
        #probabilities and adding the corresponding multiplexor
        for m in range(nqbits):
            #Calculates Conditional Probability
            conditional_probability = left_conditional_probability(
                m, probability_array)
            #Rotation angles: length: 2^(i-1)-1 and i the number of
            #qbits of the step
            thetas = 2.0*(np.arccos(np.sqrt(conditional_probability)))
            if m == 0:
                # In the first iteration it is only needed a RY gate
                qrout.apply(RY(thetas[0]), reg[0])
            else:
                # In the following iterations we have to apply
                # multiplexors controlled by m qubits
                # We call a function to construct the multiplexor,
                # whose action is a block diagonal matrix of Ry gates
                # with angles theta
                multiplexor_ry_m(qrout, reg, thetas, m, m)
        return qrout
    return p_gate_qm()

def load_r_gate(function_array):
    """
    Creates a customized AbstractGate for loading the integral of a
    discretized function in a Quantum State using Quantum Multiplexors.
    Parameters
    ----------
    function_array : numpy array
        Numpy array with the discretized function to load.
        The number of qbits will be log2(len(function_array))+1.
        Integral will be load in the last qbit
    Raises
    ----------
    AssertionError
        if len(function_array) != 2^n
    Returns
    ----------
    R_Gate: AbstractGate
        AbstractGate customized for loading the integral of the function
        using Quantum Multiplexors
    """
    text_str = 'The image of the function must be less than 1.'\
    'Rescaling is required'
    assert np.all(function_array <= 1.), text_str
    text_str = 'The image of the function must be greater than 0.'\
    'Rescaling is required'
    assert np.all(function_array >= 0.), text_str
    text_str = 'the output of the function p must be a numpy array'
    assert isinstance(function_array, np.ndarray), text_str
    nqbits = test_bins(function_array, text='Function')
    #Calculation of the rotation angles
    thetas = 2.0*np.arcsin(np.sqrt(function_array))

    @build_gate("R_Gate", [], arity=nqbits+1)
    def r_gate_qm():
        """
        Function generator for creating an AbstractGate that allows
        the loading of the integral of a given discretized function
        array into a Quantum State using Quantum Multiplexors.
        Returns
        ----------
        Qrout : quantum routine
            Routine for loading the input function as a integral
            on the last qbit.
        """
        qrout = QRoutine()
        reg = qrout.new_wires(nqbits+1)
        multiplexor_ry_m(qrout, reg, thetas, nqbits, nqbits)
        return qrout
    return r_gate_qm()
