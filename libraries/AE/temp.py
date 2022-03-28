
import numpy as np
import sys
sys.path.append("../../")

from libraries.AE.real_quantum_ae import RQAE
from libraries.utils.data_extracting import get_results
from libraries.utils.utils import bitfield_to_int
import libraries.DL.data_loading as dl

import qat.lang.AQASM as qlm
from qat.qpus import get_default_qpu
from qat.core.console import display

n = 3
x = np.arange(2**n)
p = x/np.sum(x)


oracle = qlm.QRoutine()
register = oracle.new_wires(n)
oracle.apply(dl.load_probability(p),register)

target = [0,0,1]
index = [0,1,2]

rqae = RQAE(oracle,target,index)
amplitude_estimation = rqae.run()
print(np.sqrt(p[bitfield_to_int(target)]))

qpu = get_default_qpu()


#test = shifted_oracle
#program = qlm.Program()
#qubits = program.qalloc(test.arity)
#program.apply(test,qubits)
#circuit = program.to_circ()
#display(circuit,2)


