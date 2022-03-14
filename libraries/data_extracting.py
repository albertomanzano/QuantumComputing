"""
This project has received funding from the European Union’s Horizon 2020
research and innovation programme under Grant Agreement No. 951821
https://www.neasqc.eu/

This module contains auxiliar functions for executing QLM programs based
on QLM QRoutines or QLM gates and for postproccessing results from QLM
qpu executions

Authors: Alberto Pedro Manzano Herrero & Gonzalo Ferro Costas
"""

import qat.lang.AQASM as qlm
from qat.core import Result
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(suppress=True)

def get_results(quantum_object, linalg_qpu, shots:int = 0, qubits: list = None):
    """
    Function for testing an input gate. This fucntion creates the
    quantum program for an input gate, the correspondent circuit
    and job. Execute the job and gets the results

    Parameters
    ----------
    quantum_object : QLM Gate, Routine or Program
    linalg_qpu : QLM solver
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
    dict_job = {'amp_threshold': 0.0}
    job = circuit.to_job(nbshots=shots, qubits=qubits, **dict_job)
    
    result = linalg_qpu.submit(job)
    if not isinstance(result,Result):
        result = result.join()
    
    pdf = pd.DataFrame({
            'Probability':[sample.probability for sample in result],
            'States':[sample.state for sample in result],
            'Amplitude':[sample.amplitude for sample in result],
            'Int':[sample.state.int for sample in result],
            'Int_lsb':[sample.state.lsb_int for sample in result]
        })
    pdf.reset_index(drop=True, inplace=True)
    
    pdf.sort_values('Int_lsb', inplace=True)
    return pdf, circuit, q_prog, job








