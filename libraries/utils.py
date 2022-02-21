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


#Lo de Zalo
"""
Auxiliary functions
Author: Gonzalo Ferro Costas
Version: Initial version

MyQLM version:

"""
import numpy as np
import pandas as pd

def test_bins(array, text='probability'):
    """
    Testing condition for numpy arrays. The length of the array must
    be 2^n with n an int.
    Parameters
    ----------

    array : np.ndarray
        Numpy Array whose dimensionality is going to test
    test : str
        String for identification purpouses
    Raises
    ----------

    AssertionError
        If lengt of array is not 2^n with n an int.
    Returns
    ----------

    nqbits : int
        Minimum number of qbits mandatory for storing input array in a
        quantum state
    """
    nqbits_ = np.log2(len(array))
    condition = (nqbits_%2 == 0) or (nqbits_%2 == 1)
    condition_str = 'Length of the {} Array must be of dimension 2^n with n \
        an int. In this case is: {}.'.format(text, nqbits_)
    assert condition, condition_str
    nqbits = int(nqbits_)
    return nqbits

def left_conditional_probability(initial_bins, probability):
    """
    This function calculate f(i) according to the Lov Grover and Terry
    Rudolph 2008 papper:
        'Creating superpositions that correspond to efficiently integrable
        probability distributions'
        http://arXiv.org/abs/quant-ph/0208112v1

    Given a discretized probability and an initial number of bins
    the function splits each initial region in 2 equally regions and
    calculates the condicional probabilities for x is located in the
    left part of the new regions when x is located in the region that
    contains the corresponding left region
    Parameters
    ----------

    initial_bins : int
        Number of initial bins for spliting the input probabilities
    probability : np.darray.
        Numpy array with the probabilities to be load.
        initial_bins <= len(Probability)
    Returns
    ----------

    left_cond_prob : np.darray
        conditional probabilities of the new initial_bins+1 splits
    """
    #Initial domain division
    domain_divisions = 2**(initial_bins)
    if domain_divisions >= len(probability):
        raise ValueError('The number of Initial Regions (2**initial_bins) must\
            be lower than len(probability)')
    #Original number of bins of the probability distribution
    nbins = len(probability)
    #Number of Original bins in each one of the bins of Initial
    #domain division
    bins_by_dd = nbins//domain_divisions
    #probability for x located in each one of the bins of Initial
    #domain division
    prob4dd = [
        sum(probability[j*bins_by_dd:j*bins_by_dd+bins_by_dd]) \
        for j in range(domain_divisions)
    ]
    #Each bin of Initial domain division is splitted in 2 equal parts
    bins4_left_dd = nbins//(2**(initial_bins+1))
    #probability for x located in the left bin of the new splits
    left_probabilities = [
        sum(probability[j*bins_by_dd:j*bins_by_dd+bins4_left_dd])\
        for j in range(domain_divisions)
    ]
    #Conditional probability of x located in the left bin when x is located
    #in the bin of the initial domain division that contains the split
    #Basically this is the f(j) function of the article with
    #j=0,1,2,...2^(i-1)-1 and i the number of qbits of the initial
    #domain division
    left_cond_prob = np.array(left_probabilities)/np.array(prob4dd)
    return left_cond_prob

def get_histogram(p, a, b, nbin):
    """
    Given a function p, convert it into a histogram. The function must
    be positive, the normalization is automatic. Note that instead of
    having an analytical expression, p could just create an arbitrary
    vector of the right dimensions and positive amplitudes.
    This procedure could be used to initialize any quantum state
    with real amplitudes
    Parameters
    ----------

    a : float
        lower limit of the interval
    b : float
        upper limit of the interval
    p : function
        function that we want to convert to a probability mass function
        It does not have to be normalized but must be positive
        in the interval
    nbin : int
        number of bins in the interval
    Returns
    ----------

    centers : np.darray
        numpy array with the centers of the bins of the histogtram
    probs : np.darray
        numpy array with the probability at the centers of the bins
        of the histogtram
    """
    step = (b-a)/nbin
    #Center of the bin calculation
    centers = np.array([a+step*(i+1/2) for i in range(nbin)])
    prob_n = p(centers)
    assert np.all(prob_n >= 0.), 'Probabilities must be positive, so p must be \
         a positive function'
    probs = prob_n/np.sum(prob_n)
    assert np.isclose(np.sum(probs), 1.), 'probability is not getting \
        normalized properly'
    return centers, probs

def postprocess_results(results):
    """
    Post-processing the results of simulation of a quantum circuit
    Parameters
    ----------

    results : result object from a simulation of a quantum circuit
    Parameters
    ----------

    pdf : pandas datasframe
        results of the simulation. There are 3 different columns:
        States: posible quantum basis states
        Probability: probabilities of the different states
        Amplitude: amplitude of the different states
    """
    q_probability = []
    q_states = []
    q_amplitude = []
    for sample in results:
        q_probability.append(sample.probability)
        #q_states.append(str(sample.state))
        q_states.append(sample.state)
        q_amplitude.append(sample.amplitude)
    q_probability = pd.Series(q_probability, name='Probability')
    q_states = pd.Series(q_states, name='States')
    q_amplitude = pd.Series(q_amplitude, name='Amplitude')
    pdf = pd.concat([q_states, q_probability, q_amplitude], axis=1)
    return pdf

def run_job(result):
    try:
        return result.join()
        #State = PostProcessresults(result.join())
    except AttributeError:
        return result
        #State = PostProcessresults(result)

