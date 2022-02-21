import qat.lang.AQASM as qlm

# Loading uniform distribution
@qlm.build_gate("UD", [int], arity=lambda x: x)
def uniform_distribution(number_qubits: int):
    routine = qlm.QRoutine()
    quantum_register = routine.new_wires(number_qubits)
    for i in range(number_qubits):
        routine.apply(qlm.H,quantum_register[i])
    return routine


        

        
        
        
        
        
        
        
        routine.apply(self.mask(i,axis = 0),quantum_register)
        
        return self.add_instruction(routine,add)

