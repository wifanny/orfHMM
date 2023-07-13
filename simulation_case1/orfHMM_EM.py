#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import modules
from math import inf
from math import log
import numpy as np
from scipy.stats import nbinom
from collections import defaultdict
from scipy.optimize import minimize
import random


# In[ ]:





# # log func

# In[ ]:


def log_func(number):
    '''
    convert to log form
    number: scalar
    output: scalar
    '''
    if number > 0:
        return log(number)
    else:
        return -inf


# In[ ]:





# # log-sum-exp

# In[ ]:


def logSumExp(x):
    '''
    HMM Notes p17
    log sum(exp(x1)+exp(x2)+...)
    x: list
    result: scalar
    '''
    m = max(x)
    if m == -inf: # infinity
        return -inf
    else:
        minus_m = np.array(x) - m
        result = m + log(sum(np.exp(minus_m)))
        return result   


# In[ ]:


# examples
# x = list(range(-100,-50))
# log(sum(np.exp(x)))
# logSumExp(x)


# In[ ]:





# # log likelihood f(x)

# In[ ]:


def lnNB(x, alpha, beta, E):
    '''
    Notes p2
    x: scalar. Realization from NB distribution
    alpha, beta: scalar. Parameters from Gamma distribution
    E: scalar. Normalization factor for n-th sequence
    result: scalar. Log likelihood f(x)
    '''
    r = alpha
    p = beta / (E+beta)
    result = nbinom.logpmf(x, r, p) # x follows NB(alpha, beta/(E+beta))
    return result


# In[ ]:


# examples
#x = 50
#alpha = 1
#beta = 0.1
#E = 10
#lnNB(x, alpha, beta, E)


# In[ ]:





# # Forward

# In[ ]:


def start_codon_false(i, prob_current, prob_next, observed_data, alpha_list, beta_list, E, model1):
    '''
    Calculate forward algorithm given the next codon won't start 
    i: integer. Current i-th element inside this RNA sequence
    prob_current: a list of 21 or 10 probabilites. Indicates current forward algorithm
    prob_next: a list of 21 or 10 probabilites. Indicates next forward algorithm, updating this value
    observed_data: a list. Indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 or 10 states (NB parameter)
    beta_list: a list of beta values for 21 or 10 states (NB parameter)
    E: scalar. Normalization factor for this specific sequences
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)
    output: prob_next
    '''
    if model1 == True:
        # state 1 to state 1
        temp = prob_current[0] +  log(1) + lnNB(observed_data[i+1], alpha_list[0], beta_list[0], E) 
        prob_next[0] = temp

        # state 10 and state 11 to state 11     
        log_10_11 = prob_current[9] + log(1) 
        log_11_11 = prob_current[10] + log(1) 
        temp = [log_10_11, log_11_11]
        prob_next[10] = logSumExp(temp) + lnNB(observed_data[i+1], alpha_list[10], beta_list[10], E)
        
    elif model1 == False:
        # state 1 and 10 to state 1
        log_1_1 = prob_current[0] + log_func(1)
        log_10_1 = prob_current[9] + log_func(1)
        temp = [log_1_1, log_10_1]
        prob_next[0] = logSumExp(temp) + lnNB(observed_data[i+1], alpha_list[0], beta_list[0], E)
        
    
    return prob_next


# In[ ]:


def start_codon_true(i, prob_current, prob_next, trans, next_codon, observed_data, alpha_list, beta_list, E, model1):
    '''
    Calculate forward algorithm given the next codon is one of the start codons 
    i: integer. Current i-th element inside this RNA sequence
    prob_current: a list of 21 or 10 probabilites. Indicates current forward algorithm
    prob_next: a list of 21  or 10probabilites. Indicates next forward algorithm, updating this value
    trans: a dictionary that key is the start codon (string), value is a list of scalars (three transition probability)
    next_codon: string. Indicates the specific start codon
    observed_data: a list. Indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 or 10 states (NB parameter)
    beta_list: a list of beta values for 21 or 10 states (NB parameter)
    E: scalar. Normalization factor for this specific sequences
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)
    output: prob_next
    '''
    if model1 == True:
        # state 1 to state 1
        log_1_1 = log_func(1 - trans[next_codon][0] - trans[next_codon][1])
        temp = prob_current[0] + log_1_1  + lnNB(observed_data[i+1], alpha_list[0], beta_list[0], E) 
        prob_next[0] = temp     

        # state 1 to state 2
        log_1_2 = log_func(trans[next_codon][0])
        temp = prob_current[0] + log_1_2 + lnNB(observed_data[i+1], alpha_list[1], beta_list[1], E)
        prob_next[1] = temp


        # state 1 and state 11 to state 12 
        log_1_12 = prob_current[0] + log_func(trans[next_codon][1])
        log_11_12 = prob_current[10] + log_func(trans[next_codon][2])
        temp = [log_1_12, log_11_12]   
        prob_next[11] = logSumExp(temp) + lnNB(observed_data[i+1], alpha_list[11], beta_list[11], E)

        # state 10 and state 11 to state 11     
        log_10_11 = prob_current[9] + log(1) 
        log_11_11 = prob_current[10] + log_func(1 - trans[next_codon][2])
        temp = [log_10_11, log_11_11]
        prob_next[10] = logSumExp(temp) + lnNB(observed_data[i+1], alpha_list[10], beta_list[10], E)
    
    elif model1 == False:
        # state 1 and 10 to state 1
        log_1_1 = prob_current[0] + log_func(1 - trans[next_codon])
        log_10_1 = prob_current[9] + log_func(1)
        temp = [log_1_1, log_10_1]
        prob_next[0] = logSumExp(temp) + lnNB(observed_data[i+1], alpha_list[0], beta_list[0], E)

        # state 1 to state 2
        log_1_2 = log_func(trans[next_codon])
        temp = prob_current[0] + log_1_2 + lnNB(observed_data[i+1], alpha_list[1], beta_list[1], E)
        prob_next[1] = temp
    
    return prob_next


# In[ ]:


def stop_codon_false(i, prob_current, prob_next, observed_data, alpha_list, beta_list, E, model1):
    '''
    Calculate forward algorithm given the next codon won't stop
    i: integer. Current i-th element inside this RNA sequence
    prob_current: a list of 21 or 10 probabilites. Indicates current forward algorithm
    prob_next: a list of 21 or 10 probabilites. Indicates next forward algorithm, updating this value
    observed_data: a list. Indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 or 10 states (NB parameter)
    beta_list: a list of beta values for 21 or 10 states (NB parameter)
    E: scalar. Normalization factor for this specific sequences
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)
    output: prob_next
    '''
    if model1 == True:
        # state 4 and state 7 to state 5
        log_4_5 = prob_current[3] + log(1) 
        log_7_5 = prob_current[6] + log(1)
        temp = [log_4_5, log_7_5]
        prob_next[4] = logSumExp(temp) + lnNB(observed_data[i+1], alpha_list[4], beta_list[4], E)

        # state 14 and state 17 to state 15
        log_14_15 = prob_current[13] + log(1) 
        log_17_15 = prob_current[16] + log(1) 
        temp = [log_14_15, log_17_15]
        prob_next[14] = logSumExp(temp) + lnNB(observed_data[i+1], alpha_list[14], beta_list[14], E)
    
    elif model1 == False:
        # state 4 and state 7 to state 5
        log_4_5 = prob_current[3] + log(1) 
        log_7_5 = prob_current[6] + log(1)
        temp = [log_4_5, log_7_5]
        prob_next[4] = logSumExp(temp) + lnNB(observed_data[i+1], alpha_list[4], beta_list[4], E)
    
    return prob_next


# In[ ]:


def stop_codon_true(i, prob_current, prob_next, observed_data, alpha_list, beta_list, E, model1):
    '''
    Calculate forward algorithm given the next codon is one of the stop codons 
    i: integer. Current i-th element inside this RNA sequence
    prob_current: a list of 21 or 10 probabilites. Indicates current forward algorithm
    prob_next: a list of 21 or 10 probabilites. Indicates next forward algorithm, updating this value
    observed_data: a list. Indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 or 10 states (NB parameter)
    beta_list: a list of beta values for 21 or 10 states (NB parameter)
    E: scalar. Normalization factor for this specific sequences
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)
    output: prob_next
    '''
    if model1 == True:
        # state 4 to state 5
        #prob_next[4] = prob_current[3] + log(1) + lnNB(observed_data[i+1], alpha_list[4], beta_list[4], E)

         # state 7 to state 8
        prob_next[7] = prob_current[6] + log(1) + lnNB(observed_data[i+1], alpha_list[7], beta_list[7], E)

        # state 14 to 15
       # prob_next[14] = prob_current[13] + log(1) + lnNB(observed_data[i+1], alpha_list[14], beta_list[14], E)

        # state 17 to state 18
        prob_next[17] = prob_current[16] + log(1) + lnNB(observed_data[i+1], alpha_list[17], beta_list[17], E)
    
    elif model1 == False:
        # state 4 to state 5
        prob_next[4] = prob_current[3] + log(1) + lnNB(observed_data[i+1], alpha_list[4], beta_list[4], E)

        # state 7 to state 8
        prob_next[7] = prob_current[6] + log(1) + lnNB(observed_data[i+1], alpha_list[7], beta_list[7], E)
    
    return prob_next


# In[ ]:


def forward_algorithm(RNA_data, observed_data, alpha_list, beta_list, E, trans, stop_codon_list, num_sequence, model1):
    '''
    Compute and store the forward algorithm for a given sequence
    RNA_data: a list of lists. Each inner list indicates a single RNA sequence and this list contains letters 'A', 'C', 'U', 'G'
    observed_data: a list of lists. Each inner list indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 or 10 states (NB parameter)
    beta_list: a list of beta values for 21 or 10 states (NB parameter)
    E: a list of scalars. Normalization factor for all sequences
    trans: a dictionary that key is the start codon (string), value is a list of scalars (three transition probability)
    stop_codon_list: a list of stop codons (string)
    num_sequence: scalar indicates n-th sequence
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)
    output: matrix stores the forward algorithm for single RNA sequence
    '''

    # Find E, observed_data, RNA_data according to the order of sequences
    E = E[num_sequence - 1]
    observed_data = observed_data[num_sequence - 1]
    RNA_data = RNA_data[num_sequence - 1]
    
    sequence_length = len(RNA_data) # length of data
    if model1 == True:
        output = np.zeros((sequence_length, 21)) # initialize output matrix
        prob_current = np.ones(21) * (-inf) # initialize probability, 21 different states
    elif model1 == False:
        output = np.zeros((sequence_length, 10))
        prob_current = np.ones(10) * (-inf) # initialize probability, 10 different states
    prob_current[0] = lnNB(observed_data[0], alpha_list[0], beta_list[0], E)
    output[0] = prob_current

    
    for i in range(sequence_length - 1):
        start_codon = False
        stop_codon = False
        next_codon = ""
        if model1 == True:
            prob_next = np.ones(21) * (-inf)
        elif model1 == False:
            prob_next = np.ones(10) * (-inf)
        
        # Check start and stop codon
        if i + 3 < sequence_length:
            next_codon = RNA_data[i+1] + RNA_data[i+2] + RNA_data[i+3] 
            if next_codon in trans.keys():
                start_codon = True
            
            if next_codon in stop_codon_list:
                stop_codon = True
           

        # start codon is false
        if start_codon == False:               
            prob_next = start_codon_false(i, prob_current, prob_next, observed_data, alpha_list, beta_list, E, model1)
        
        # start codon is ture
        elif start_codon == True:   
            prob_next = start_codon_true(i, prob_current, prob_next, trans, next_codon, observed_data, 
                                           alpha_list, beta_list, E, model1)
        
        # stop codon is ture
        if stop_codon == True: 
            prob_next = stop_codon_true(i, prob_current, prob_next, observed_data, alpha_list, beta_list, E, model1)
 
        # stop codon is false
        elif stop_codon == False:
            prob_next = stop_codon_false(i, prob_current, prob_next, observed_data, alpha_list, beta_list, E, model1)
       
        # transter to next state with probability 1
        if model1 == True:
            sure_to_transit = [1, 2, 4, 5, 7, 8, 11, 12, 14, 15, 17, 18]
            #sure_to_transit = [1, 2, 3, 4, 5, 7, 8, 11, 12,13, 14, 15, 17, 18]
        elif model1 == False:
            sure_to_transit = [1, 2, 4, 5, 7, 8]
        for k in sure_to_transit:
            temp = prob_current[k] + log(1) + lnNB(observed_data[i+1], alpha_list[k+1], beta_list[k+1], E)
            prob_next[k+1] = temp

        if model1 == True:
            # state 20 and state 21 to state 21     
            log_20_21 = prob_current[19] + log(1) 
            log_21_21 = prob_current[20] + log(1) 
            temp = [log_20_21, log_21_21]
            prob_next[20] = logSumExp(temp) + lnNB(observed_data[i+1], alpha_list[20], beta_list[20], E)
    
        prob_current = prob_next
        output[i+1] = prob_current
  
    return output


# In[ ]:


def forward_matrix(RNA_data, observed_data, alpha_list, beta_list, E, trans, stop_codon_list, model1):
    '''
    Compute forward matrix by combining result from each single sequence
    RNA_data: a list of lists. Each inner list indicates a single RNA sequence and this list contains letters 'A', 'C', 'U', 'G'
    observed_data: a list of lists. Each inner list indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 or 10 states (NB parameter)
    beta_list: a list of beta values for 21 or 10 states (NB parameter)
    E: a list of scalars. Normalization factor for all sequences
    trans: a dictionary that key is the start codon (string), value is a list of scalars (three transition probability)
    stop_codon_list: a list of stop codons (string)
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)
    output: matrix stores the forward algorithm for multiple RNA sequences
    '''
    output = []
    for n in range(1, len(observed_data)+1):
        output.append(forward_algorithm(RNA_data, observed_data, alpha_list, beta_list, E, trans, stop_codon_list, n, model1))
    return output


# In[ ]:


# forward = forward_matrix(RNA_data, observed_data, alpha_list, beta_list, E, trans, stop_codon_list)


# # Backward

# In[ ]:


def start_codon_false_back(i, prob_current_back, prob_previous_back, observed_data, alpha_list, beta_list, E, model1):
    '''
    Calculate backward algorithm given the next codon won't start 
    i: integer. Current i-th element inside this RNA sequence
    prob_current_back: a list of 21 or 10 probabilites. Indicates current backward algorithm
    prob_previous_back: a list of 21 or 10 probabilites. Indicates previous backward algorithm, updating this value
    observed_data: a list. Indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 or 10 states (NB parameter)
    beta_list: a list of beta values for 21 or 10 states (NB parameter)
    E: scalar. Normalization factor for this specific sequences
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)
    output: prob_previous_back
    '''
    
    # state 1 to state 1
    temp = prob_current_back[0] + log(1) + lnNB(observed_data[i+1], alpha_list[0], beta_list[0], E) 
    prob_previous_back[0] = temp
 
    if model1 == True:
        # state 11 to state 11     
        temp = prob_current_back[10] + log(1) + lnNB(observed_data[i+1], alpha_list[10], beta_list[10], E)
        prob_previous_back[10] = temp
    
    return prob_previous_back


# In[ ]:


def start_codon_true_back(i, prob_current_back, prob_previous_back, trans, next_codon, observed_data, alpha_list, beta_list, E, model1):
    '''
    Calculate backward algorithm given the next codon is one of the start codons 
    i: integer. Current i-th element inside this RNA sequence
    prob_current_back: a list of 21 or 10 probabilites. Indicates current backward algorithm
    prob_previous_back: a list of 21 or 10 probabilites. Indicates previous backward algorithm, updating this value
    trans: a dictionary that key is the start codon (string), value is a list of scalars (three transition probability)
    next_codon: string. Indicates the specific start codon
    observed_data: a list. Indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 or 10 states (NB parameter)
    beta_list: a list of beta values for 21 or 10 states (NB parameter)
    E: scalar. Normalization factor for this specific sequences
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)
    output: prob_previous_back
    '''
    if model1 == True:
        # state 1 to state 1
        temp = log_func(1 - trans[next_codon][0] - trans[next_codon][1])
        log_1_1 = prob_current_back[0] + temp + lnNB(observed_data[i+1], alpha_list[0], beta_list[0], E)

        # state 1 to state 2
        temp = log_func(trans[next_codon][0])
        log_1_2 = prob_current_back[1] + temp + lnNB(observed_data[i+1], alpha_list[1], beta_list[1], E) 

        # state 1 to state 12
        temp = log_func(trans[next_codon][1]) 
        log_1_12 = prob_current_back[11] + temp + lnNB(observed_data[i+1], alpha_list[11], beta_list[11], E) 

        temp = [log_1_1, log_1_2, log_1_12]
        prob_previous_back[0] = logSumExp(temp)


        # state 11 to state 11, 12
        temp = log_func(1 - trans[next_codon][2])
        log_11_11 = prob_current_back[10] + temp + lnNB(observed_data[i+1], alpha_list[10], beta_list[10], E) 

        temp = log_func(trans[next_codon][2])
        log_11_12 = prob_current_back[11] + temp + lnNB(observed_data[i+1], alpha_list[11], beta_list[11], E) 

        temp = [log_11_11, log_11_12]
        prob_previous_back[10] = logSumExp(temp)
        
        
    elif model1 == False:
        # state 1 to state 1, 2
        temp = log_func(1 - trans[next_codon])
        log_1_1 = prob_current_back[0] + temp + lnNB(observed_data[i+1], alpha_list[0], beta_list[0], E) 

        temp = log_func(trans[next_codon])
        log_1_2 = prob_current_back[1] + temp + lnNB(observed_data[i+1], alpha_list[1], beta_list[1], E) 

        temp = [log_1_1, log_1_2]
        prob_previous_back[0] = logSumExp(temp)
    
    return prob_previous_back


# In[ ]:


def stop_codon_false_back(i, prob_current_back, prob_previous_back, observed_data, alpha_list, beta_list, E, model1):
    '''
    Calculate backward algorithm given the next codon won't stop
    i: integer. Current i-th element inside this RNA sequence
    prob_current_back: a list of 21 or 10 probabilites. Indicates current backward algorithm
    prob_previous_back: a list of 21 or 10 probabilites. Indicates previous backward algorithm, updating this value
    observed_data: a list. Indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 or 10 states (NB parameter)
    beta_list: a list of beta values for 21 or 10 states (NB parameter)
    E: scalar. Normalization factor for this specific sequences
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)
    output: prob_previous_back
    '''
    
    # state 7 to state 5
    prob_previous_back[6] = prob_current_back[4] + log(1) + lnNB(observed_data[i+1], alpha_list[4], beta_list[4], E)
      
    if model1 == True:
        # state 17 to state 15
        prob_previous_back[16] = prob_current_back[14] + log(1) + lnNB(observed_data[i+1], alpha_list[14], beta_list[14], E)

    return prob_previous_back


# In[ ]:


def stop_codon_true_back(i, prob_current_back, prob_previous_back, observed_data, alpha_list, beta_list, E, model1):
    '''
    Calculate backward algorithm given the next codon is stop codon
    i: integer. Current i-th element inside this RNA sequence
    prob_current_back: a list of 21 or 10 probabilites. Indicates current backward algorithm
    prob_previous_back: a list of 21 or 10 probabilites. Indicates previous backward algorithm, updating this value
    observed_data: a list. Indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 or 10 states (NB parameter)
    beta_list: a list of beta values for 21 or 10 states (NB parameter)
    E: scalar. Normalization factor for this specific sequences
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)
    output: prob_previous_back
    '''
    
    # state 7 to state 8
    prob_previous_back[6] = prob_current_back[7] + log(1) + lnNB(observed_data[i+1], alpha_list[7], beta_list[7], E)
    
    if model1 == True:
        # state 17 to state 18
        prob_previous_back[16] = prob_current_back[17] + log(1) + lnNB(observed_data[i+1], alpha_list[17], beta_list[17], E)
  
    return prob_previous_back


# In[ ]:


def backward_algorithm(RNA_data, observed_data, alpha_list, beta_list, E, trans, stop_codon_list, num_sequence, model1):
    '''
    Compute and store the backward algorithm for a given sequence
    RNA_data: a list of lists. Each inner list indicates a single RNA sequence and this list contains letters 'A', 'C', 'U', 'G'
    observed_data: a list of lists. Each inner list indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 or 10 states (NB parameter)
    beta_list: a list of beta values for 21 or 10 states (NB parameter)
    E: a list of scalars. Normalization factor for all sequences
    trans: a dictionary that key is the start codon (string), value is a list of scalars (three transition probability)
    stop_codon_list: a list of stop codons (string)
    num_sequence: scalar indicates n-th sequence
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)
    output: matrix stores the backward algorithm for single RNA sequence
    '''

    # Find E, observed_data, RNA_data according the order of sequences
    E = E[num_sequence - 1]
    observed_data = observed_data[num_sequence - 1]
    RNA_data = RNA_data[num_sequence - 1]
    
    sequence_length = len(RNA_data) # length of data
    if model1 == True:
        prob_current_back = list(np.zeros(21)) # initialize probability, 21 different states
        output = np.zeros((sequence_length, 21)) # initialize matrix
    elif model1 == False:
        prob_current_back = list(np.zeros(10)) # initialize probability, 10 different states
        output = np.zeros((sequence_length, 10))
    output[sequence_length - 1] = prob_current_back  
    
    iteration_list = np.arange(0, sequence_length-1, 1).tolist()
    iteration_list.reverse()
    for i in iteration_list:
        start_codon = False
        stop_codon = False
        if model1 == True:
            prob_previous_back = list(np.ones(21) * (-inf))
        elif model1 == False:
            prob_previous_back = list(np.ones(10) * (-inf))
        next_codon = ""
        
        # Check start and stop codon
        if i + 3 < sequence_length:
            next_codon = RNA_data[i+1] + RNA_data[i+2] + RNA_data[i+3]  
            if next_codon in trans.keys():
                start_codon = True
            
            if next_codon in stop_codon_list:
                stop_codon = True
        
        # start codon is false
        if start_codon == False:       
            prob_previous_back = start_codon_false_back(i, prob_current_back, prob_previous_back, observed_data, 
                                                        alpha_list, beta_list, E, model1)
        
        # start codon is true
        elif start_codon == True:   
            prob_previous_back = start_codon_true_back(i, prob_current_back, prob_previous_back, trans, 
                                                       next_codon, observed_data, alpha_list, beta_list, E, model1)
        
        # stop codon is true
        if stop_codon == True: 
            prob_previous_back = stop_codon_true_back(i, prob_current_back, prob_previous_back, observed_data, 
                                                      alpha_list, beta_list, E, model1) 
        
        # stop codon is false
        elif stop_codon == False:
            prob_previous_back = stop_codon_false_back(i, prob_current_back, prob_previous_back, observed_data, 
                                                       alpha_list, beta_list, E, model1)
     
        
        # traster with probability 1
        if model1 == True:
            sure_to_transit = [2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20]
	   
        elif model1 == False:
            sure_to_transit = [2, 3, 4, 5, 6, 8, 9]
        for k in sure_to_transit:
            temp = prob_current_back[k] + log(1) + lnNB(observed_data[i+1], alpha_list[k], beta_list[k], E)
            prob_previous_back[k-1] = temp

        if model1 == True:
            # state 21 to state 21     
            prob_previous_back[20] = prob_current_back[20] + log(1) + lnNB(observed_data[i+1], alpha_list[20], beta_list[20], E)
        elif model1 == False:
            # state 10 to state 1     
            prob_previous_back[9] = prob_current_back[0] + log(1) + lnNB(observed_data[i+1], alpha_list[0], beta_list[0], E)  
            
        prob_current_back = prob_previous_back 
        output[i] = prob_current_back
        
    return output


# In[ ]:


def backward_matrix(RNA_data, observed_data, alpha_list, beta_list, E, trans, stop_codon_list, model1):
    '''
    Compute backward matrix by combining result from each single sequence
    RNA_data: a list of lists. Each inner list indicates a single RNA sequence and this list contains letters 'A', 'C', 'U', 'G'
    observed_data: a list of lists. Each inner list indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 or 10 states (NB parameter)
    beta_list: a list of beta values for 21 or 10 states (NB parameter)
    E: a list of scalars. Normalization factor for all sequences
    trans: a dictionary that key is the start codon (string), value is a list of scalars (three transition probability)
    stop_codon_list: a list of stop codons (string)
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)
    output: matrix stores the backward algorithm for multiple RNA sequences
    '''
    output = []
    for n in range(1, len(observed_data)+1):
        output.append(backward_algorithm(RNA_data, observed_data, alpha_list, beta_list, E, trans, 
                                         stop_codon_list, n, model1))
    return output


# In[ ]:


#backward =  backward_matrix(RNA_data, observed_data, alpha_list, beta_list, E, trans, stop_codon_list)


# In[ ]:





# # Incomplete log likelihood (simple) with backward (test accuracy)

# In[ ]:


#def incomplete_log_likelihood(backward, observed_data, alpha_list, beta_list, E):
    '''
    Notes p3
    Compute the incomplete log likelihood with backward algorithm
    backward: a list of matrix. Stores backward algorithm output
    observed_data: a list of lists. Each inner list indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 or 10 states (NB parameter)
    beta_list: a list of beta values for 21 or 10 states (NB parameter)
    E: a list of scalars. Normalization factor for all sequences
    output: scalar. Indicates incomplete log likelihood
    '''
   
    # initialize parameters
    #temp_list = []
    
    # incomplete log likelihood
    #for n in range(1, len(observed_data)+1):
        
     #   B1_1 = backward[n-1][0][0]
     #   nb = lnNB(observed_data[n-1][0], alpha_list[0], beta_list[0], E[n-1])
     #   temp_list.append(B1_1 + nb)
       
   # output = logSumExp(temp_list)
        
   # return output


# In[ ]:


#incomplete_log_likelihood(backward, observed_data, alpha_list, beta_list, E)


# In[ ]:





# # Incomplete log likelihood (simple) with forward (actual use)

# In[ ]:


def incomplete_log_likelihood(forward, observed_data, alpha_list, beta_list, E):
    '''
    Notes p3
    Compute the incomplete log likelihood with forward algorithm
    backward: a list of matrix. Stores backward algorithm output
    observed_data: a list of lists. Each inner list indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 or 10 states (NB parameter)
    beta_list: a list of beta values for 21 or 10 states (NB parameter)
    E: a list of scalars. Normalization factor for all sequences
    output: scalar. Indicates incomplete log likelihood
    '''
    # initialize parameters
    output_list = []
    #  incomplete log likelihood
    for n in range(1, len(observed_data)+1):
    
        last_position = len(observed_data[n-1]) - 1
    #    print(forward[n-1][last_position])
        output_list.append(logSumExp(forward[n-1][last_position]))
         
    return np.sum(output_list)


# In[ ]:


#incomplete_log_likelihood(forward, observed_data, alpha_list, beta_list, E)


# In[ ]:





# # Compute L(simple)

# In[ ]:


def compute_L(forward, backward, model1):
    '''
    Notes p4
    Compute one state probability L 
    forward: a list of matrix. Compute and store by forward algorithm
    backward: a list of matrix. Compute and store by backward algorithm
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)
    output: a list of matrix
    '''
    
    output = []

    # n sequences
    for n in range(1, len(forward)+1):
        if model1 == True:
            sequence = np.zeros((len(forward[n-1]), 21))
        elif model1 == False:
            sequence = np.zeros((len(forward[n-1]), 10))

        for t in range(1, len(forward[n-1])+1):
            
            temp = logSumExp((forward[n-1] + backward[n-1])[t-1]) # denominator 
                   
            sequence[t-1] = forward[n-1][t-1] + backward[n-1][t-1] - temp
            
        output.append(sequence)
    return output


# In[ ]:


#L = compute_L(forward, backward)


# # Compute H

# In[ ]:


def transprob_comp(curr_state, next_state, codon_specific, trans, model1):
    '''
    Compute transition probability given the current state and next state with specific codon
    curr_state: int. Indicates the current state (1-21 or 1-10)
    next_state: int. Indicates the next state (1-21 or 1-10 )
    codon_specific: string. Indicates the specific codon
    trans: dictionary
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)
    output: scalar
    '''
    if model1 == True:
        if curr_state == 1:
            if next_state == 1:
                return log_func(1 - trans[codon_specific][0] - trans[codon_specific][1])
            elif next_state == 2:
                return log_func(trans[codon_specific][0])
            elif next_state == 12:
                return log_func(trans[codon_specific][1])
        if curr_state == 11:
            if next_state == 11:
                return log_func(1 - trans[codon_specific][2])
            elif next_state == 12:
                return log_func(trans[codon_specific][2])
            
    if model1 == False:
        if curr_state == 1:
            if next_state == 1:
                return log_func(1 - trans[codon_specific])
            elif next_state == 2:
                return log_func(trans[codon_specific])
   


# In[ ]:


def compute_H_nume(forward, backward, curr_state, next_state, codon_specific, observed_data, alpha_list, beta_list, num_sequence, t, trans, E, model1):
    '''
    Notes p4
    Compute numerator for H function
    forward: a list of matrix
    backward: a list of matrix
    curr_state: int. Indicates the current state (1-21 or 1-10)
    next_state: int. Indicates the next state (1-21 or 1-10)
    codon_specific: string. Indicates the specific codon
    observed_data: a list of lists. Indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 or 10 states (NB parameter)
    beta_list: a list of beta values for 21 or 10 states (NB parameter)
    num_sequence: int. Indicates which RNA sequence
    t: int. Indicates the position inside RNA sequence
    trans: dictionary
    E: list. Normalization factor
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)
    output: scalar
    '''
    transprob = transprob_comp(curr_state, next_state, codon_specific, trans, model1)
    numerator = forward[num_sequence - 1][t - 1][curr_state - 1] +                transprob + lnNB(observed_data[num_sequence - 1][t], alpha_list[next_state - 1], beta_list[next_state - 1], E[num_sequence - 1]) +                 backward[num_sequence - 1][t][next_state - 1]
    return numerator


# In[ ]:


def compute_H_deno(forward, backward, curr_state, codon_specific, observed_data, alpha_list, beta_list, num_sequence, t, trans, E, model1):
    '''
    Notes p4
    Compute denominator for H function according to current state
    forward: a list of matrix
    backward: a list of matrix
    curr_state: int. Indicates the current state (1-21)
    next_state: int. Indicates the next state (1-21)
    codon_specific: string. Indicates the specific codon
    observed_data: a list. Indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 or 10 states (NB parameter)
    beta_list: a list of beta values for 21 or 10 states (NB parameter)
    num_sequence: int. Indicates which RNA sequence
    t: int. Indicates the position inside RNA sequence
    trans: dictionary
    E: list. Normalization factor
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)
    output: scalar
    '''
    output_list = [] 
        
    # case state1 to state1, 2, 12
    if curr_state == 1:
        if model1 == True:
            temp = [1, 2, 12]
        elif model1 == False:
            temp = [1, 2]
        for next_state in temp:
            output_list.append(compute_H_nume(forward, backward, curr_state, next_state, codon_specific, 
                                              observed_data, alpha_list, beta_list, num_sequence, t, trans, 
                                              E, model1))
    
    # case state11 to state11, 12
    elif curr_state == 11:
        for next_state in [11, 12]:
            output_list.append(compute_H_nume(forward, backward, curr_state, next_state, codon_specific, 
                                              observed_data, alpha_list, beta_list, num_sequence, t, trans, 
                                              E, model1))

    return logSumExp(output_list)
        


# In[ ]:


def compute_H_codon(forward, backward, curr_state, next_state, codon_specific, RNA_data, observed_data, alpha_list, beta_list, E, trans, t, num_sequence, model1):
    '''
    Notes p4
    Compute codon specific H function 
    forward: a list of matrix
    backward: a list of matrix
    curr_state: int. Indicates the current state (1-21 or 1-10)
    next_state: int. Indicates the next state (1-21 or 1-10)
    codon_specific: string. Indicates the specific codon
    RNA_data: list of lists
    observed_data: a list of lists. Indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 or 10 states (NB parameter)
    beta_list: a list of beta values for 21 or 10 states (NB parameter)
    E: a list of scalars
    trans: dictionary
    t: int. Indicates the position inside RNA sequence
    num_sequence: int. Indicates which RNA sequence
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)
    output: scalar
    '''
    # initialize parameters
    output = -inf
    numerator_list = []
    # check codon
    if t + 3 <= len(RNA_data[num_sequence - 1]):
        next_codon = RNA_data[num_sequence - 1][t] + RNA_data[num_sequence - 1][t+1] + RNA_data[num_sequence - 1][t+2]  
        
        if codon_specific == next_codon:
   
            numerator = compute_H_nume(forward, backward, curr_state, next_state, codon_specific, observed_data, 
                                       alpha_list, beta_list, num_sequence, t, trans, E, model1)
            if numerator == -inf:
                return -inf
            
    
            denominator = compute_H_deno(forward, backward, curr_state, codon_specific, observed_data, 
                                         alpha_list, beta_list, num_sequence, t, trans, E, model1)
        
            output = numerator - denominator
            
            
    return output


# # store probabilities

# In[ ]:



def store_info(forward, backward, RNA_data, observed_data, alpha_list, beta_list, E, trans, stop_codon_list, ci, model1):
    '''
    Store codon specific H function 
    forward: a list of matrix
    backward: a list of matrix
    RNA_data: list of lists
    observed_data: a list of lists. Indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 states (NB parameter)
    beta_list: a list of beta values for 21 states (NB parameter)
    E: a list of scalars
    trans: dictionary
    stop_codon_list: list of stop codons (strings)
    ci: string. Given codon
    model1: boolean (True/False). Identify it's model1 (21-states) or model2 (10-states)
    output: a list of scalars
    '''
    temp = []
    
        
    n11_list = []
    n12_list = []
    n112_list = []
    n1111_list = []
    n1112_list = []
    for n in range(1, len(observed_data) + 1):
            
        for t in range(1, len(observed_data[n-1]) - 1):
            
            n11_list.append(compute_H_codon(forward, backward, 1, 1, ci, RNA_data, observed_data, 
                                            alpha_list, beta_list, E, trans, t, n, model1))
            n12_list.append(compute_H_codon(forward, backward, 1, 2, ci, RNA_data, observed_data, 
                                            alpha_list, beta_list, E, trans, t, n, model1))
            if model1 == True:
                n112_list.append(compute_H_codon(forward, backward, 1, 12, ci, RNA_data, observed_data, 
                                                 alpha_list, beta_list, E, trans, t, n, model1))
                n1111_list.append(compute_H_codon(forward, backward, 11, 11, ci, RNA_data, observed_data, 
                                                  alpha_list, beta_list, E, trans, t, n, model1))
                n1112_list.append(compute_H_codon(forward, backward, 11, 12, ci, RNA_data, observed_data, 
                                                  alpha_list, beta_list, E, trans, t, n, model1))
  
    n11 = logSumExp(n11_list)
    n12 = logSumExp(n12_list)
    if model1 == True:
        n112 = logSumExp(n112_list)
        n1111 = logSumExp(n1111_list)
        n1112 = logSumExp(n1112_list)
        
        
    temp.append(np.exp(n11))
    temp.append(np.exp(n12))
    if model1 == True:
        temp.append(np.exp(n112))
        temp.append(np.exp(n1111))
        temp.append(np.exp(n1112))
        
    return temp
                


# # three parameters

# In[ ]:


def update_three_parameters(forward, backward, RNA_data, observed_data, alpha_list, beta_list, E, trans, stop_codon_list, model1):
    '''
    Update transition probability 
    forward: a list of matrix
    backward: a list of matrix
    RNA_data: list of lists
    observed_data: a list of lists. Indicates the height of a sequence and this list contains scalars
    alpha_list: a list of alpha values for 21 states (NB parameter)
    beta_list: a list of beta values for 21 states (NB parameter)
    E: a list of scalars
    trans: dictionary
    stop_codon_list: list of stop codons (strings)
    model1: boolean (True/False). Identify it's model1 (21-states) or model2 (10-states)
    output: dictionary, similar to trans
    '''
    output = defaultdict(list)
    for ci in trans.keys():
        
        info = store_info(forward, backward, RNA_data, observed_data, alpha_list, beta_list, E, trans, 
                          stop_codon_list, ci, model1)
        
        # it it's 21-states model
        if model1 == True:
            deno_alpha = info[0]+info[1]+info[2]
            deno_delta = info[3] + info[4]
            output[ci] = [info[1]/deno_alpha, info[2]/deno_alpha, info[4]/deno_delta]
            
        # if it's 10-states model
        else:
            deno_alpha = info[0]+info[1]
            output[ci] = info[1]/deno_alpha
        
    return output
                
    


# In[ ]:


#update_three_parameters(forward, backward, RNA_data, observed_data, alpha_list, beta_list, E, trans, stop_codon_list)


# # alpha - M

# In[ ]:


# Note p9 
#Q function with respect to alpha
def object_function_alpha(x, state, forward, observed_data, beta_list, E, L): # list of alpha_m
    
    output = 0
    
    for n in range(1, len(forward)+1):
        for t in range(1, len(forward[n-1])+1):
            temp = lnNB(observed_data[n-1][t-1], x, beta_list[state-1], E[n-1])
            output += np.exp(L[n-1][t-1][state-1]) * temp      
        
    return((-1)*output)


# In[ ]:


# derivative function with respect to alpha
def derivative_function_alpha(x, state, forward, observed_data, beta_list, E, L): # list of alpha_m
        
    output_sum = 0
    for n in range(1, len(observed_data)+1):           
              
        for t in range(1, len(observed_data[n-1])+1):
             
            sum_s = 0
            for s in range(1, observed_data[n-1][t-1]+1):
                    
                sum_s += 1/(x + observed_data[n-1][t-1] - s)
                                 
            latter = sum_s + log(beta_list[state-1] / (E[n-1] + beta_list[state-1]))

            output_sum += np.exp(L[n-1][t-1][state-1]) * latter
               
    
    return (-1)*output_sum


# In[ ]:





# In[ ]:





# # beta - M

# In[ ]:


# Note p10
# Q function with respect to beta
def object_function_beta(x, state, forward, observed_data, alpha_list, E, L): # list of beta_m
    
    output = 0
    
    for n in range(1, len(forward)+1):
        for t in range(1, len(forward[n-1])+1):
            temp = lnNB(observed_data[n-1][t-1], alpha_list[state-1], x, E[n-1])
            output += np.exp(L[n-1][t-1][state-1]) * temp
            
        
    return ((-1)*output)    


# In[ ]:


# derivative function with respect to beta
def derivative_function_beta(x, state, forward, observed_data, alpha_list, E, L): # list of beta_m
        
    output_sum = 0
    for n in range(1, len(observed_data)+1):
                 
        for t in range(1, len(observed_data[n-1])+1):
                
            first_term = alpha_list[state-1]/x
            second_term = (alpha_list[state-1]+observed_data[n-1][t-1])/(E[n-1]+x)
               
            output_sum += np.exp(L[n-1][t-1][state-1]) * (first_term - second_term)      
           
    return (-1)* output_sum


# In[ ]:





# In[ ]:





# In[ ]:





# # EM iteration

# In[ ]:


def EM_iter(RNA_data, observed_data, E, trans_init, alpha_init, beta_init, epsilon, max_iter, fixed, stop_codon_list, model1):
    '''
    Updating all parameters
    RNA_data: a list of lists. Each inner list represents a RNA sequence
    observed_data: a list of lists. Each inner list includes sclars
    E: a list of scalars. Normalization factors
    trans_init: dictionary. key: start codon, value a list of transition parameters
    alpha_init: alpha value for different states
    beta_init: beta value for different states
    epsilon: scalar. difference between two log likelihood smaller than this, then stop
    max_iter: int. max number of iteration times
    fixed: boolean (True/False). Indicates wanna beta fixed or not (False represents update both alpha and beta)
    stop_codon_list: a list of stop codons
    model1: boolean (True/False). Identify it's model1 (21-states: True) or model2 (10-states: False)
    output:
            trans: dictionary. updated transition probability for different start codons
            alpha_list: updated alpha values
            beta_list: updated beta values
    '''
    
    
    trans = trans_init.copy()
    alpha_list = alpha_init.copy()
    beta_list = beta_init.copy()
    
    # compute initial forward and backward
    forward = forward_matrix(RNA_data, observed_data, alpha_list, beta_list, E, trans, stop_codon_list, model1)
    backward = backward_matrix(RNA_data, observed_data, alpha_list, beta_list, E, trans, stop_codon_list, model1)  
    
    log_links = []
    log_links.append(incomplete_log_likelihood(forward, observed_data, alpha_init, beta_init, E))
    ##print(log_links) ##remove
    
    delta = 1
    n_iter = 1
    while((np.abs(delta) > epsilon) and (n_iter < max_iter)):
        ##print("iteration" + str(n_iter)) ##remove
        
        curr_trans = trans.copy()
        curr_alpha_list = alpha_list.copy()
        if fixed == False:
            curr_beta_list = beta_list.copy()
       
        L = compute_L(forward, backward, model1)
        
        # update rho_u_ci, rho_ci, delta_ci
        trans = update_three_parameters(forward, backward, RNA_data, observed_data, alpha_list, beta_list, E, trans, stop_codon_list, model1)
        ##print(trans) ##remove
        
        # order of updating 
        if model1 == True:
            state_list = np.arange(21)+1
        elif model1 == False:
            state_list = np.arange(10)+1
        random.shuffle(state_list)
        
        # update alpha_list and beta_list
        for i in state_list:
            state = i
           
            ##print("state"+str(state)) ##remove
            res = minimize(object_function_alpha, alpha_list[state-1], method='BFGS', tol = 1, jac=derivative_function_alpha, args = (state, forward, observed_data, beta_list, E, L), options={'disp': False, 'maxiter': 10,'gtol': 1})
            alpha_list[state-1] = res.x[0]
            ##print("alpha"+str(res.x[0])) ##remove
        #    print(res.success)        
           
        for i in state_list:
            state = i             
            if fixed == False:
                res = minimize(object_function_beta, beta_list[state-1], method='nelder-mead', tol = 1, args = (state, forward, observed_data, alpha_list, E, L), options={'disp': False})
                #res = minimize(object_function_beta, beta_list[state-1], method='BFGS', tol = 1, jac=derivative_function_beta, args = (state, forward, observed_data, curr_alpha_list, E, L), options={'disp': False, 'maxiter': 10,'gtol': 1})
                beta_list[state-1] = res.x[0]
        #        print(res.success)
                ##print("beta"+str(res.x[0])) ##remove
        
        ##print(alpha_list) ##remove
        ##print(beta_list) ##remove

        # update forwad and backward
        forward = forward_matrix(RNA_data, observed_data, alpha_list, beta_list, E, trans, stop_codon_list, model1)
        backward = backward_matrix(RNA_data, observed_data, alpha_list, beta_list, E, trans, stop_codon_list, model1)  
        
        # log likelihood
        log_links.append(incomplete_log_likelihood(forward, observed_data, alpha_list, beta_list, E))
        delta = log_links[-1] - log_links[-2]
        n_iter += 1
        
        ##print(log_links) ##remove

    debug = forward    
    if fixed == False:
        return (trans, alpha_list, beta_list, log_links, debug)
    else:
        return (trans, alpha_list, log_links)


# In[ ]:





# In[ ]:




