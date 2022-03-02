import qat.lang.AQASM as qlm
from qat.qpus import LinAlg
from qat.core import Result
import numpy as np

# Measure probabilities
def measure_probabilities(gate,shots = 0):
    program = qlm.Program()
    number_qubits = gate.arity
    quantum_register = program.qalloc(number_qubits)
    program.apply(gate,quantum_register)
    circuit = program.to_circ()
    
    
    job = circuit.to_job(nbshots = shots)
    linalg_qpu = LinAlg()
    result = linalg_qpu.submit(job)
    if not isinstance(result,Result):
        result = result.join()

    # Result
    result_array = np.zeros(2**number_qubits)
    for sample in result:
        result_array[sample.state.lsb_int] = sample.probability
    return result_array

# Measure probabilities
def measure_amplitudes(gate,shots = 0):
    program = qlm.Program()
    quantum_register = program.qalloc(gate.arity)
    program.apply(gate,quantum_register)
    circuit = program.to_circ()

    job = circuit.to_job(nbshots = shots)
    linalg_qpu = LinAlg()
    result = linalg_qpu.submit(job)
    if not isinstance(result,Result):
        result = result.join()

    # Result
    result_array = np.zeros(2**number_qubits)
    for sample in result:
        result_array[sample.state.lsb_int] = sample.probability
    return result_array