import qat.lang.AQASM as qlm
import numpy as np

# Convierte un entero n en un array de bits de longitud size
def bitfield(n,size):
    aux = [1 if digit=='1' else 0 for digit in bin(n)[2:]]
    right = np.array(aux)
    left = np.zeros(max(size-right.size,0))    
    full = np.concatenate((left, right))
    return full.astype(int)

@qlm.build_gate("Mask", [int,int], arity=lambda x,y: x)
def mask(number_qubits,index):
    routine = qlm.QRoutine()
    quantum_register = routine.new_wires(number_qubits)
    bits = bitfield(index,number_qubits)
    for k in range(number_qubits):
        if bits[-k-1] == 0:
            routine.apply(qlm.X,quantum_register[k])

    return routine






















