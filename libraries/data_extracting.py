import qat.lang.AQASM as qlm
#from qat.qpus import LinAlg
from qat.core import Result
import numpy as np
from utils import run_job, postprocess_results
from copy import deepcopy

def create_qprogram(quantum_gate):
    """
    Creates a Quantum Program from an input qlm gate or routine

    Parameters
    ----------

    quantum_gate : QLM gate or QLM routine

    Returns
    ----------
    q_prog: QLM Program.
        Quantum Program from input QLM gate or routine
    """
    q_prog = qlm.Program()
    qbits = q_prog.qalloc(quantum_gate.arity)
    q_prog.apply(quantum_gate, qbits)
    return q_prog

def get_results(quantum_object, lineal_qpu, shots=0, qubits=None):
    """
    Function for testing an input gate. This fucntion creates the
    quantum program for an input gate, the correspondent circuit
    and job. Execute the job and gets the results

    Parameters
    ----------
    quantum_object : QLM Gate, Routine or Program
    lineal_qpu : QLM solver
    shots : int 
        number of shots for the generated job.
        if 0 True probabilities will be computed
    qubits : list
        list with the qbits for doing the measurement when simulating
        if None measuremnt over all allocated qbits will be provided

    Returns
    ----------
    pdf_ : pandas DataFrame
        DataFrame with the results of the simulation
    circuit : QLM circuit
    q_prog : QLM Program.
    job : QLM job

    """
    if type(quantum_object) == qlm.Program:
        q_prog = deepcopy(quantum_object)
    else:
        q_prog = qlm.Program()
        qbits = q_prog.qalloc(quantum_object.arity)
        q_prog.apply(quantum_object, qbits)
    circuit = q_prog.to_circ(submatrices_only=True)
    if qubits is None:
        job = circuit.to_job(nbshots=shots)
    if type(qubits) == list:
        job = circuit.to_job(nbshots=shots, qubits=qubits)
    result = run_job(lineal_qpu.submit(job))
    pdf_ = postprocess_results(result)
    return pdf_, circuit, q_prog, job













