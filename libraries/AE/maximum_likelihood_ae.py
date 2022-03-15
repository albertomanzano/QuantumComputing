"""
Copyright 2022 CESGA
License:

This project has received funding from the European Union’s Horizon 2020
research and innovation programme under Grant Agreement No. 951821
https://www.neasqc.eu/

This module contains necesary functions and classes to implement
Maximum Likelihood Amplitude Estimation based on the paper:

    Suzuki, Y., Uno, S., Raymond, R., Tanaka, T., Onodera, T., & Yamamoto, N.
    Amplitude estimation without phase estimation
    Quantum Information Processing, 19(2), 2020
    arXiv: quant-ph/1904.10246v2

Author:Gonzalo Ferro Costas

MyQLM version:

"""
import sys
sys.path.append("../../")

import copy
import numpy as np
import pandas as pd
import scipy.optimize as so

import qat.lang.AQASM as qlm 
from qat.qpus import get_default_qpu

from AE.amplitude_amplification import grover
from data_extracting import get_results
from libraries.utils import bitfield_to_int
from copy import deepcopy



class MLAE:
    """
    Class for using Maximum Likelihood Quantum Amplitude Estimation (ML-AE)
    algorithm
    """

    def __init__(self,oracle: qlm.QRoutine,target: list,index: list,**kwargs):
        """

        Method for initializing the class
    
        Parameters
        ----------
        
        kwars : dictionary
            dictionary that allows the configuration of the MLAE algorithm:
            Implemented keys:
            oracle: QLM gate
                QLM gate with the Oracle for implementing the
                Groover-like operator:
                init_q_prog and q_gate will be interpreted as None
            list_of_mks : list
                python list with the different m_ks for executing the algortihm
            nbshots : int
                number of shots for quantum job. If 0 exact probabilities
                will be computed.
            qpu : QLM solver
                solver for simulating the resulting circutis
        """
        #Setting attributes
        self.grover_oracle = None
        self.target = target
        self.index = index    
        self.oracle = oracle

        #Set the QPU to use
        self.linalg_qpu = kwargs.get('qpu')
        if (self.linalg_qpu is None):
            self.linalg_qpu = get_default_qpu()

        # The schedule of the method
        self.m_k = kwargs.get('m_k', [10])
        self.n_k = kwargs.get('n_k',[100]*len(self.m_k))
        self.h_k = [0]*len(self.m_k)
        self.compute_h_k()

        # Optimization
        self.optimizer = kwargs.get('optimizer',\
                lambda x: so.brute(func = x,ranges = [(0,np.pi/2)]))


    @property
    def oracle(self):
        return self._oracle

    @oracle.setter
    def oracle(self,value):
        self._oracle = deepcopy(value) 
        self.grover_oracle = grover(self.oracle,self.target,self.index)
    
    @property
    def m_k(self):
        return self._m_k

    @m_k.setter
    def m_k(self, value):
        if type(value) in [list, int]:
            if type(value) in [int]:
                self._m_k = list(range(value))
                self._m_k.reverse()
            else:
                self._m_k = value
        else:
            raise ValueError('For m_k only ints and list are aceptable types')
        #We update the allocate classical bits each time we change cbits_number
    
    @property
    def n_k(self):
        return self._n_k

    @n_k.setter
    def n_k(self,value):
        if (type(value)==int) and (value>0):
            self._n_k = [value]*len(self.m_k)
        elif (type(value)==list):
            if (len(value)==len(self.m_k)):
                self._n_k = value
            else:
                raise ValueError("Sizes of m_k and nbshots must match. Otherwise pass an int")
        else:
            self._n_k = [100]*len(self.m_k)
        

 
    def run_step(self,m_k: int,n_k: int) -> int:
        """
        This method applies the Grover-like operator self.q_gate to the
        Quantum Program self.q_prog a given number of times m_k.

        Parameters
        ----------
        m_k : int
            number of times to apply the self.q_gate to the quantum circuit

        Returns
        ----------

        pdf_mks : pandas dataframe
            results of the measurement of the last qbit
        circuit : QLM circuit object
            circuit object generated for the quantum program
        job : QLM job object
        """
        routine = qlm.QRoutine()
        register = routine.new_wires(self.oracle.arity)
        routine.apply(self.oracle,register)
        for i in range(m_k):
            routine.apply(self.grover_oracle,register)
        result, circuit, _, job = get_results(routine,linalg_qpu = self.linalg_qpu,shots = n_k,qubits = self.index)
        h_k = int(result["Probability"].iloc[bitfield_to_int(self.target)]*n_k)

        return h_k


    @staticmethod
    def likelihood(theta: float, m_k: int, n_k: int,h_k: int)->float:
        """
        Calculates Likelihood from Suzuki papper. For h_k positive events
        of n_k total events this function calculates the probability of
        this taking into account that the probability of a positive
        event is given by theta and by m_k
        The idea is use this function to minimize it for this reason it gives
        minus Likelihood

        Parameters
        ----------

        theta : float
            Angle (radians) for calculating the probability of measure a
            positive event.
        m_k : pandas Series
            For MLQAE this a pandas Series where each row is the number of
            times the operator Q was applied.
            We needed for calculating the probability of a positive event
            for eack posible m_k value: sin((2*m_k+1)theta)**2.
        h_k : pandas Series
            Pandas Series where each row is the number of positive events
            measured for each m_k
        n_k : pandas Series
            Pandas Series where each row is the number of total events
            measured for each m_k

        Returns
        ----------

        float
            Gives the -Likelihood of the inputs

        """
        with np.errstate(divide='ignore'):
            theta_ = (2*m_k+1)*theta
            first_term = 2*h_k*np.log(np.abs(np.sin(theta_)))
            second_term = 2*(n_k-h_k)*np.log(np.abs(np.cos(theta_)))
            l_k = first_term + second_term
        return l_k

    def compute_h_k(self):
        for i in range(len(self.m_k)):
            self.h_k[i] = self.run_step(self.m_k[i],self.n_k[i])

    def cost_function(self,angle: float)->float:
        """
        This method calculates the Likelihood for theta between [0, pi/2]
        for an input pandas DataFrame.

        Parameters
        ----------

        pdf_input: pandas DataFrame
            The DataFrame should have following columns:
            m_k: number of times q_gate was applied
            h_k: number of times the state |1> was measured
            n_k: number of total measuremnts

        N : int
            number of division for the theta interval

        Returns
        ----------

        y : pandas DataFrame
            Dataframe with the likelihood for the p_mks atribute. It has
            2 columns:
            theta : posible valores of the angle
            l_k : likelihood for each theta

        """
        cost = 0
        for i in range(len(self.m_k)):
            cost+=MLAE.likelihood(angle,self.m_k[i],self.n_k[i],self.h_k[i])

        return -cost
    
    def optimize(self)->float:
        """
        This functions execute a brute force optimization of the
        likelihood function for an input results pdf.

        Parameters
        ----------

        results : pandas DataFrame
            DataFrame with the results from ml-qpe procedure.
            Mandatory columns:
            m_k : number of times Groover like operator was applied
            h_k : number of measures of the state |1>
            n_k : number of measurements done


        Returns
        ----------

        optimum_theta : float
            theta  that minimize likelihood
        """
        result = self.optimizer(self.cost_function)
        return result[0]
