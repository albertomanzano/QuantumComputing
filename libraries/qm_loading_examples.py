"""
Authors: Juan Santos Suárez & Gonzalo Ferro Costas
Version: Initial version

MyQLM version:

This script contains several examples of use of the gates implemented
in the QuantumMultiplexors_Module
"""

import numpy as np
from utils import postprocess_results, run_job, get_histogram
from data_loading import load_p_gate, load_r_gate
from qat.lang.AQASM import Program, H
from qat.core.console import display

def load_probability_program(p_x):
    """
    Creates a Quantum Program for loading an input numpy array with a
    probability distribution with Quantum Multiplexors (QM).

    Parameters
    ----------

    p_x : numpy array
        Probability distribution of size m. Mandatory: m=2^n where n
        is the number qbits of the quantum circuit.

    Returns
    ----------
    q_prog: QLM Program.
        Quantum Program for loading input probability using QM
    p_gate: QLM AbstractGate
        Customized AbstractGate for loading input probability using QM
    """
    p_gate = load_p_gate(p_x)
    q_prog = Program()
    qbits = q_prog.qalloc(p_gate.arity)
    q_prog.apply(p_gate, qbits)
    return q_prog, p_gate

def load_integral_program(f_x):
    """
    Creates a Quantum Program for loading the integral of an input
    function given as a numpy array using Quantum Multiplexors (QM).

    Parameters
    ----------

    f_x : numpy array
        Function evaluation of size m. Mandatory: m=2^n where n is the
        number qbits of the quantum circuit.

    Returns
    ----------
    q_prog: QLM Program
        Quantum Program for loading integral of the input function
    r_gate: QLM AbstractGate
        Customized AbstractGate for loading integral using QM
    """
    r_gate = load_r_gate(f_x)
    q_prog = Program()
    qbits = q_prog.qalloc(r_gate.arity)
    for i in range(len(qbits)-1):
        q_prog.apply(H, qbits[i])
    q_prog.apply(r_gate, qbits)
    return q_prog, r_gate

def expectation_loading_data(p_x, f_x):
    """
    Creates a Quantum Program for loading mandatory data in order to
    load the expected value of a function f(x) over a x following a
    probability distribution p(x) using Quantum Multiplexors (QM).

    Parameters
    ----------

    p_x : numpy array
        Probability distribution of size m. Mandatory: m=2^n where n
        is the number qbits of the quantum circuit.
    f_x : numpy array
        Function evaluation of size m. Mandatory: m=2^n where n is the
        number qbits of the quantum circuit.

    Returns
    ----------
    q_prog: QLM Program.
        Quantum Program for loading input probability
    p_gate: QLM AbstractGate
        Customized AbstractGate for loading input probability using QM
    r_gate: QLM AbstractGate
        Customized AbstractGate for loading integral using QM
    """

    #Testing input
    assert len(p_x) == len(f_x), 'Arrays lenght are not equal!!'
    p_gate = load_p_gate(p_x)
    r_gate = load_r_gate(f_x)
    q_prog = Program()
    #The R gate have more qbits
    qbits = q_prog.qalloc(r_gate.arity)
    #Load Probability
    q_prog.apply(p_gate, qbits[:-1])
    #Load integral on the last qbit
    q_prog.apply(r_gate, qbits)
    return q_prog, p_gate, r_gate


def Do(n_qbits=6, depth=0, function='DataLoading'):
    """
    Function for testing purpouses. This function is used when the
    script is executed from command line using arguments. It executes
    the three implemented fucntions of this script:
        * LoadProbabilityProgram
        * LoadIntegralProgram
        * ExpectationLoadingData

    Parameters
    ----------
    n_qbits : int.
        Number of Qbits for the quantum circuit.
    depth : int
        Depth for visualizar the Quantum Circuit
    function : str
        String that indicates which of the before functions should be
        used:
            'P' : LoadProbabilityProgram
            'I' : LoadIntegralProgram
            Otherwise : ExpectationLoadingData
    """
    def p(x):
        return x*x
    def f(x):
        return np.sin(x)
    #The number of bins
    m_bins = 2**n_qbits
    lower_limit = 0.0
    upper_limit = 1.0
    x, p_x = get_histogram(p, lower_limit, upper_limit, m_bins)
    f_x = f(x)
    print('########################################')
    print('#########Connection to QLMaSS###########')
    print('########################################')

    #QPU connection
    try:
        from qat.qlmaas import QLMaaSConnection
        connection = QLMaaSConnection('qlm')
        lin_alg = connection.get_qpu("qat.qpus:LinAlg")
        lineal_qpu = lin_alg()
    except (ImportError, OSError) as e:
        print('Problem: usin PyLinalg')
        from qat.qpus import PyLinalg
        lineal_qpu = PyLinalg()

    print('Creating Program')
    if function == 'P':
        print('\t Load Probability')
        qprog, _ = load_probability_program(p_x)
    elif function == 'R':
        print('\t Load Integral')
        qprog, _ = load_integral_program(f_x)
    else:
        print('\t Load Complete Data')
        qprog, _, _ = expectation_loading_data(p_x, f_x)

    print('Making Circuit')
    circuit = qprog.to_circ(submatrices_only=True)
    display(circuit, max_depth=depth)
    if function == 'P':
        job = circuit.to_job()
    else:
        job = circuit.to_job(qubits=[n_qbits])
    result = run_job(lineal_qpu.submit(job))
    results = postprocess_results(result)
    print(results)
    if function == 'P':
        condition = np.isclose(results['Probability'], p_x).all()
        print('Probability load data: \n {}'.format(p_x))
        print('Probability Measurements: \n {}'.format(results['Probability']))
        print('This is correct? {}'.format(condition))
    elif function == 'R':
        integral_measurement = results['Probability'][1]*2**(n_qbits)
        print('Integral load data: {}'.format(sum(f_x)))
        print('Integral Measurement: {}'.format(integral_measurement))
        condition = np.isclose(integral_measurement, sum(f_x))
        print('This is correct? {}'.format(condition))
    else:
        integral_measurement = results['Probability'][1]
        print('Integral Measurement: {}'.format(integral_measurement))
        print('Expectation of f(x) for x~p(x): Integral p(x)f(x):\
            {}'.format(sum(p_x*f_x)))
        condition = np.isclose(integral_measurement, sum(p_x*f_x))
        print('This is correct? {}'.format(condition))


if __name__ == '__main__':
    #Working Example
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--nqbits',
        type=int,
        help='Number Of Qbits', default=6
    )
    parser.add_argument(
        '-depth',
        type=int,
        help='Depth of the Diagram',
        default=0
    )
    parser.add_argument(
        '-t',
        '--type',
        default=None,
        help='Type of Loading: P: Load Probability. R: Load Integral.\
        Otherwise: Load Complete Data'
    )
    args = parser.parse_args()
    #print(args)

    Do(n_qbits=args.nqbits, depth=args.depth, function=args.type)


