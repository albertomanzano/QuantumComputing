"""
Copyright 2022 CESGA
This project has received funding from the European Union’s Horizon 2020
research and innovation programme under Grant Agreement No. 951821
https://www.neasqc.eu/

This module contains all functions needed for creating Grover-like
operators mandatory for using Quantum Amplitude Amplification
and Estimation as explained in the 2000 Brassard paper:

    Gilles Brassard, Peter Hoyer, Michele Mosca and Alain Tapp
    Quantum Amplitude Amplification and Estimation
    AMS Contemporary Mathematics Series, 305, 06-2000
    https://arxiv.org/abs/quant-ph/0005055v1


Version: Initial version

MyQLM version:

"""
import numpy as np
import time
import qat.lang.AQASM as qlm 
from copy import deepcopy


def reflection(lista: np.ndarray):
    number_qubits = len(lista)
    @qlm.build_gate("R_{"+str(lista)+"}", [], arity= number_qubits)
    def reflection_gate():
        routine = qlm.QRoutine()
        register = routine.new_wires(number_qubits)

        for i in range(number_qubits):
            if (lista[i] == 0):
                routine.apply(qlm.X,register[-i-1])
        routine.apply(qlm.Z.ctrl(len(lista)-1),register)
        for i in range(number_qubits):
            if (lista[i] == 0):
                routine.apply(qlm.X,register[-i-1])
        return routine
    return reflection_gate()



def U0(oracle: qlm.QRoutine,target: np.ndarray,index: np.ndarray):
    number_qubits = oracle.arity
    @qlm.build_gate("U_0_"+str(time.time_ns()), [], arity=number_qubits)
    def U0_gate():
    #def U0_gate(oracle):
        routine = qlm.QRoutine()
        register = routine.new_wires(number_qubits)
        routine.apply(reflection(target),[register[i] for i in index])
        return routine
    #return U0_gate(oracle)
    return U0_gate()


def U(oracle: qlm.QRoutine):
    oracle_cp = deepcopy(oracle)
    number_qubits = oracle.arity
    #@qlm.build_gate("U", [object], arity=number_qubits)
    @qlm.build_gate("U_"+str(time.time_ns()), [], arity=number_qubits)
    def U_gate():   
    #def U_gate(oracle_cp):   
        routine = qlm.QRoutine()
        register = routine.new_wires(number_qubits)
        routine.apply(oracle.dag(),register)
        routine.apply(reflection(np.zeros(number_qubits,dtype=int)),register)
        routine.apply(oracle,register)
        return routine
    return U_gate()
    #return U_gate(oracle_cp)


def grover(oracle: qlm.QRoutine,target: np.ndarray,index: np.ndarray):
    oracle_cp = deepcopy(oracle)
    number_qubits = oracle_cp.arity
    #@qlm.build_gate("G", [object], arity=number_qubits)
    @qlm.build_gate("G_"+str(time.time_ns()), [], arity=number_qubits)
    def grover_gate():        
    #def grover_gate(oracle_cp):        
        routine = qlm.QRoutine()
        register = routine.new_wires(number_qubits)
        routine.apply(U0(oracle_cp,target,index),register)
        routine.apply(U(oracle_cp),register)
        return routine
    return grover_gate()
    #return grover_gate(oracle_cp)




