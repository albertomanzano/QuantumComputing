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
import copy
import numpy as np
import pandas as pd
import scipy.optimize as so

import qat.lang.AQASM as qlm 
from qat.qpus import get_default_qpu

from AE.amplitude_amplification import grover
from data_extracting import get_results


def likelihood(theta, m_k, h_k, n_k):
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
    theta_ = (2*m_k+1)*theta
    first_term = 2*h_k*np.log(np.abs(np.sin(theta_)))
    second_term = 2*(n_k-h_k)*np.log(np.abs(np.cos(theta_)))
    l_k = first_term + second_term
    return -np.sum(l_k)

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
            dictionary that allows the configuration of the ML-QPE algorithm:
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
            delta : float
                For avoiding problems when calculating the domain for theta
            default_nbshots : int
                default number of measurements for computing freqcuencies
                when nbshots for quantum job is 0
            iterations : int
                number of iterations of the optimizer
            display : bool
                for displaying additional information in the optimization step
            initial_state : QLM Program
                QLM Program withe the initial Psi state over the
                Grover-like operator will be applied
                Only used if oracle is None
        """
        #Setting attributes
        self.oracle = copy.deepcopy(oracle)
        self.target = target
        self.index = index
            
        #Creates the Grover-like operator from oracle
        self.grover_oracle = grover(self.oracle,target,index)


        #A complete list of m_k
        self.list_of_mks_ = kwargs.get('list_of_mks', 10)
        self.list_of_mks = self.list_of_mks_

        #If 0 we compute the exact probabilities
        self.shots = kwargs.get('nbshots', 0)
        #Set the QPU to use
        self.linalg_qpu = kwargs.get('qpu')
        if (self.linalg_qpu is None):
            self.linalg_qpu = get_default_qpu
        ##delta for avoid problems in 0 and pi/2 theta limits
        self.delta = kwargs.get('delta', 1.0e-5)
        #This is the default number of shots used for computing
        #the freqcuencies of the results when the computed probabilities
        #instead of freqcuencies are provided (nbshots = 0 when qlm job
        #is created)
        self.default_nbshots = kwargs.get('default_nbshots', 100)
        #number of iterations for optimization of Likelihood
        self.iterations = kwargs.get('iterations', 100)
        #For displaying extra info for optimization proccess
        self.disp = kwargs.get('disp', True)
        #Setting attributes
        self.restart()


    @property
    def list_of_mks(self):
        return self.list_of_mks_

    @list_of_mks.setter
    def list_of_mks(self, value):
        if type(value) in [list, int]:
            if type(value) in [int]:
                self.list_of_mks_ = list(range(value))
                self.list_of_mks_.reverse()
            else:
                self.list_of_mks_ = value
            print('list_of_mks: {}'.format(self.list_of_mks))

        else:
            raise ValueError('For m_k only ints and list are aceptable types')
        #We update the allocate classical bits each time we change cbits_number

 
    def run_step(self, m_k):
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
        result, circuit, _, job = get_results(routine,linalg_qpu = self.linalg_qpu,shots = self.shots,qubits = self.index)

        return result["Probability"]


    def run_mlae(self):#, list_of_mks=None, nbshots=None):
        """
        This method is the core of the Maximum Likelihood Amplitude
        Estimation. It runs several quantum circuits each one increasing
        the number of self.q_gate applied to the the initial self.q_prog
    
        Parameters
        ----------
        
        list_of_mks : list (corresponding property will be overwrite)
            python list with the different m_ks for executing the algortihm
        nbshots : int (corresponding property will be overwrite)
            number of shots for quantum job. If 0 exact probabilities
            will be computed

        """

        self.restart()
        #Clean the list in each run
        for m_k in self.list_of_mks:
            self.run_step(m_k)

        self.theta = self.launch_optimizer(self.pdf_mks)


    def launch_likelihood(self, pdf_input, N=100):
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
        pdf = pdf_input.copy(deep=True)
        if pdf is None:
            print(
                """
                Can not calculate Likelihood because pdf_input is empty.
                Please provide a valida DataFrame.
                """)
            return None
        theta = np.linspace(0+self.delta, 0.5*np.pi-self.delta, N)
        m_k = pdf['m_k']
        h_k = pdf['h_k']
        n_k = pdf['n_k']
        l_k = np.array([likelihood(t, m_k, h_k, n_k) for t in theta])
        y_ = pd.DataFrame({'theta': theta, 'l_k': l_k})
        return y_

    def launch_optimizer(self, results):
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

        theta_domain = (0+self.delta, 0.5*np.pi-self.delta)
        optimizer = so.brute(
            likelihood,
            [theta_domain],
            (results['m_k'], results['h_k'], results['n_k']),
            self.iterations,
            disp=self.disp
        )
        optimum_theta = optimizer[0]
        return optimum_theta

