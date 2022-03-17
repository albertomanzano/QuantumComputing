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

from copy import deepcopy
import numpy as np
import pandas as pd
import scipy.optimize as so

import qat.lang.AQASM as qlm 
from qat.qpus import get_default_qpu

from libraries.AE.amplitude_amplification import grover
from libraries.data_extracting import get_results
from libraries.utils import bitfield_to_int, check_list_type



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
        self._oracle = deepcopy(oracle)
        self._target = check_list_type(target,int)
        self._index = check_list_type(index,int)
        self._grover_oracle = grover(self._oracle,self.target,self.index)

        #Set the QPU to use
        self.linalg_qpu = kwargs.get('qpu')
        if (self.linalg_qpu is None):
            self.linalg_qpu = get_default_qpu()

        # The schedule of the method
        self.m_k = None
        self.n_k = None
        self.h_k = None
        schedule = kwargs.get('schedule',None)
        if (schedule is None):
            self.set_linear_schedule(5,10)
        else:
            self.schedule = schedule

        # Optimization
        self.optimizer = kwargs.get('optimizer',\
                lambda x: so.brute(func = x,ranges = [(0,np.pi/2)]))

    #####################################################################
    @property
    def oracle(self):
        return self._oracle

    @oracle.setter
    def oracle(self,value):
        self._oracle = deepcopy(value) 
        self._grover_oracle = grover(self.oracle,self.target,self.index)
   
    @property
    def target(self):
        return self._target

    @target.setter
    def target(self,value):
        self._target = check_list_type(value,int)
        self._grover_oracle = grover(self.oracle,self.target,self.index)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self,value):
        self._index = check_list_type(value,int)
        self._grover_oracle = grover(self.oracle,self.target,self.index)

    @property
    def schedule(self):
        return self._schedule

    @schedule.setter
    def schedule(self,value):
        x = check_list_type(value,int)
        if (x.shape[0]!=2):
            raise Exception("The shape of the schedule must be (2,n)")
        self._schedule = x
        self.m_k = self.schedule[0]
        self.n_k = self.schedule[1]        
    #####################################################################

    def set_linear_schedule(self,n: int,n_k: int):
        x = np.arange(n)
        y = [n_k]*n
        self.schedule = [x,y]
    
    def set_exponential_schedule(self,n: int,n_k: int):
        x = 2**np.arange(n)
        y = [n_k]*n
        self.schedule = [x,y]

        
 
    def run_step(self,m_k: int,n_k: int) -> int:
        """
        This method executes the Grover-like operator self._grover_oracle to the
        a given number of times m_k.

        Parameters
        ----------
        m_k : int
            number of times to apply the self.q_gate to the quantum circuit
        n_k : int
            number of shots 

        Returns
        ----------
        h_k: int
            number of positive events
        """
        routine = qlm.QRoutine()
        register = routine.new_wires(self.oracle.arity)
        routine.apply(self.oracle,register)
        for i in range(m_k):
            routine.apply(self._grover_oracle,register)
        result, circuit, _, job = get_results(routine,linalg_qpu = self.linalg_qpu,shots = n_k,qubits = self.index)
        h_k = int(result["Probability"].iloc[bitfield_to_int(self.target)]*n_k)

        return h_k


    @staticmethod
    def likelihood(theta: float, m_k: int, n_k: int,h_k: int)->float:
        """
        Calculates Likelihood from Suzuki papper. For h_k positive events
        of n_k total events, this function calculates the probability of
        this taking into account that the probability of a positive
        event is given by theta and by m_k
        The idea is use this function to minimize it for this reason it gives
        minus Likelihood

        Parameters
        ----------

        theta : float
            Angle (radians) for calculating the probability of measure a
            positive event.
        m_k : int
            number of times the grover operator was applied.
        n_k : int
            number of total events measured for the specific  m_k
        h_k : int
            number of positive events measured for each m_k

        Returns
        ----------

        float
            Gives the Likelihood p(h_k with m_k amplifications|theta)

        """
        theta_ = (2*m_k+1)*theta
        p_0 = np.sin(theta_)**2
        p_1 = np.cos(theta_)**2
        l_k = (p_0**h_k)*(p_1**(n_k-h_k))
        return l_k
    
    @staticmethod
    def log_likelihood(theta: float, m_k: int, n_k: int,h_k: int)->float:
        """
        Calculates log of the likelihood from Suzuki papper.  

        Parameters
        ----------

        theta : float
            Angle (radians) for calculating the probability of measure a
            positive event.
        m_k : int
            number of times the grover operator was applied.
        n_k : int
            number of total events measured for the specific  m_k
        h_k : int
            number of positive events measured for each m_k

        Returns
        ----------

        float
            Gives the log Likelihood p(h_k with m_k amplifications|theta)

        """
        theta_ = (2*m_k+1)*theta
        p_0 = np.sin(theta_)**2
        p_1 = np.cos(theta_)**2
        l_k = h_k*p_0 + (n_k-h_k)*p_1
        return l_k


    def cost_function(self,angle: float)->float:
        """
        This method calculates the -Likelihood of angle theta
        for a given schedule m_k,n_k

        Parameters
        ----------

        angle: float
            the hypothetical angle

        Returns
        ----------

        cost : float
            the aggregation of the individual likelihoods
        """
        log_cost = 0
        for i in range(len(self.m_k)):
            log_l_k = MLAE.log_likelihood(angle,self.m_k[i],self.n_k[i],self.h_k[i])
            log_cost = log_cost+log_l_k
        return -log_cost
    
    def optimize(self)->float:
        """
        This functions execute a brute force optimization of the
        likelihood function for an input results pdf.

        Parameters
        ----------

        Returns
        ----------

        result[0] : float
            angle that minimizes the cost function
        """
        self.h_k = np.zeros(len(self.m_k),dtype = int)
        for i in range(len(self.m_k)):
            self.h_k[i] = self.run_step(self.m_k[i],self.n_k[i])
        
        result = self.optimizer(self.cost_function)
        return result
