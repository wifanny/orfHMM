# generate multiple synthetic transcripts
import numpy as np
import random

def generate_nb(alpha_list_true, beta_list_true, state, E=1):
    '''
    Given the state, generate an RPF count from a NB distribution
    
    alpha_list_true: float. true alpha values
    beta_list_true: float. true beta values
    state: int.
    E: int. transcript expression level 
    '''
    return np.random.negative_binomial(alpha_list_true[state-1], beta_list_true[state-1]/(E+beta_list_true[state-1]))

def SingleRNA(start_codon_list, transition_list, E, repeat_u, repeat, len_3UTR, len_min_5UTR, 
              len_min_5UTR2, alpha_list_true, beta_list_true):
    '''
    Simulate single RNA sequence contains 2 scenarios: both uORF and CDS, only CDS
    
    '''
    stop_codon_list = ['UAA','UGA','UAG']

    # indexs for initiation
    state_1_2 = 0
    state_1_12 = 0
    state_11_12 = 0

    # initialization
    curr_RNA = []
    observed_counts = []
    states_seq = []

    # generate bases for the beginning (not allow for the occurrence of start codons)
    for i in range(len_min_5UTR):
      if i < 3:
        temp = random.choices(['A', 'C', 'G', 'U'], weights=np.ones(4), k = 1)[0]
        curr_RNA.append(temp)
      else: 
        temp = random.choices(['A', 'C', 'G', 'U'], weights=np.ones(4), k = 1)[0]
        curr_RNA.append(temp)
        while "".join(curr_RNA[-3:]) in start_codon_list:
          temp = random.choices(['A', 'C', 'G', 'U'], weights=np.ones(4), k = 1)[0]
          curr_RNA[-1] = temp
    
    states_seq.extend([1]*(len_min_5UTR-2))

    # start from 5'UTR
    state = 1
    while state == 1: 
      temp = random.choices(['A', 'C', 'G', 'U'], weights=np.ones(4), k = 1)[0]
      curr_RNA.append(temp)

      if "".join(curr_RNA[-3:]) in start_codon_list:
        for i, item in enumerate(start_codon_list):
          if "".join(curr_RNA[-3:]) in item:
             s_index = i
        prob = transition_list[s_index]
        state = random.choices([1, 2, 12], weights=[1-transition_list[s_index][0]-transition_list[s_index][1], transition_list[s_index][0], transition_list[s_index][1]], k = 1)[0]
        if state == 1:
          states_seq.append(1)

      else:
        states_seq.append(1)

    if state == 2:
      state_1_2 = 1
    elif state == 12:
      state_1_12 = 1

    # Enter in translated uORF
    if state == 2:

      # initiation 
      states_seq.extend((2,3,4))
        
      # add bases for elongation recycling (not allow for the occurrence of stop codons)
      for i in range(repeat_u):
        triplet = random.sample(['A', 'C', 'G', 'U'], 3)
        while "".join(triplet) in stop_codon_list:
          triplet = random.sample(['A', 'C', 'G', 'U'], 3)
        curr_RNA += triplet

      states_seq.extend([5,6,7]*repeat_u)

      # termination 
      curr_RNA += random.choices([['U', 'A', 'A'], ['U', 'G', 'A'], ['U', 'A', 'G']], weights=[1, 1, 1], k = 1)[0]
      states_seq.extend((8,9,10))

      state = 11


    # Enter in 5′UTR2
    # Add bases for 5′U T R2 (not allow for the occurrence of start codons)
    if state == 11:
      for i in range(len_min_5UTR2):
        if i < 3:
          temp = random.choices(['A', 'C', 'G', 'U'], weights=np.ones(4), k = 1)[0]
          curr_RNA.append(temp)
        else: 
          temp = random.choices(['A', 'C', 'G', 'U'], weights=np.ones(4), k = 1)[0]
          curr_RNA.append(temp)
          while "".join(curr_RNA[-3:]) in start_codon_list:
            temp = random.choices(['A', 'C', 'G', 'U'], weights=np.ones(4), k = 1)[0]
            curr_RNA[-1] = temp

      states_seq.extend([11]*(len_min_5UTR2-2))

    while state == 11:
      temp = random.choices(['A', 'C', 'G', 'U'], weights=np.ones(4), k = 1)[0]
      curr_RNA.append(temp)

      if "".join(curr_RNA[-3:]) in start_codon_list:
        for i, item in enumerate(start_codon_list):
          if "".join(curr_RNA[-3:]) in item:
             s_index = i
        prob = transition_list[s_index]
        state = random.choices([11, 12], weights=[1-transition_list[s_index][2], transition_list[s_index][2]], k = 1)[0]
        if state == 11:
          states_seq.append(11)
        else:
          state_11_12 = 1

      else:
        states_seq.append(11)


   # Enter in translated main CDS
    if state == 12:

     # initiation 
      states_seq.extend((12,13,14))
        
      # add bases for elongation recycling (not allow for the occurrence of stop codons)
      for i in range(repeat):
        triplet = random.sample(['A', 'C', 'G', 'U'], 3)
        while "".join(triplet) in stop_codon_list:
          triplet = random.sample(['A', 'C', 'G', 'U'], 3)
        curr_RNA += triplet

      states_seq.extend([15,16,17]*repeat)

      # termination 
      curr_RNA += random.choices([['U', 'A', 'A'], ['U', 'G', 'A'], ['U', 'A', 'G']], weights=[1, 1, 1], k = 1)[0]
      states_seq.extend((18,19,20))


    # Enter in 3’UTR
    for i in range(len_3UTR):
      curr_RNA += random.choices(['A', 'C', 'G', 'U'], weights=np.ones(4), k = 1)[0]
    states_seq.extend([21]*len_3UTR)

    # generate the corresponding observed count sequence
    for s in states_seq:
      observed_counts.append(generate_nb(alpha_list_true, beta_list_true, s, E))

    return [curr_RNA, observed_counts, states_seq, [state_1_2, state_1_12, state_11_12]]

def simulate_main(num, start_codon_list, transition_list, E, 
                  repeat_u, repeat, len_3UTR, len_min_5UTR, len_min_5UTR2, alpha_list_true, beta_list_true):
    '''
    Simulate multiple RNA sequence contains 3 scenarios: only uORF + only CDS + both

    ''' 

    RNA_data = []
    counts_data = []
    states_data = []
    transition_index = []

    for i in range(num):
        temp = SingleRNA(start_codon_list, transition_list, E[i], repeat_u[i], 
                         repeat[i], len_3UTR[i], len_min_5UTR[i], len_min_5UTR2[i], alpha_list_true, beta_list_true)
        
        # change to only uORF
        if i%3 == 0:
          if temp[3][0] == 1 and temp[3][2] == 1:
            observed_counts = temp[1]
            states_seq = temp[2]
           
            start_index = states_seq.index(12)

            # replace original state sequence
            states_replace = [11 for i in states_seq[start_index:]]
            states_seq[start_index:] = states_replace

            # replace original count sequence
            count_replace = [generate_nb(alpha_list_true, beta_list_true, 11, E[i]) for j in observed_counts[start_index:]]
            observed_counts[start_index:] = count_replace

            temp[1] = observed_counts
            temp[2] = states_seq
            temp[3][2] = 0

        
        RNA_data.append(temp[0])
        counts_data.append(temp[1])
        states_data.append(temp[2])
        transition_index.append(temp[3])
  
    return [RNA_data, counts_data, states_data, transition_index]

#def simulate_main(num, start_codon_list, transition_list, E, 
#                   repeat_u, repeat, len_3UTR, len_min_5UTR, len_min_5UTR2, alpha_list_true, beta_list_true):
#     '''
#     Simulate multiple RNA sequence contains 3 scenarios: only uORF + only CDS + both

#     ''' 

#     RNA_data = []
#     counts_data = []
#     states_data = []
#     transition_index = []

#     for i in range(num):
#         temp = SingleRNA(start_codon_list, transition_list, E[i], repeat_u[i], 
#                          repeat[i], len_3UTR[i], len_min_5UTR[i], len_min_5UTR2[i], alpha_list_true, beta_list_true)
        
#         RNA_data.append(temp[0])
#         counts_data.append(temp[1])
#         states_data.append(temp[2])
#         transition_index.append(temp[3])
  
#     return [RNA_data, counts_data, states_data, transition_index]

def simulate_neither(num, length, E, alpha_list_true, beta_list_true):
  '''
  Simulate multiple RNA sequence without translated regions
  ''' 
  RNA_data = []
  counts_data = []
  states_data = []
  transition_index = []

  for i in range(num):
    RNA_seq = []
    observed_counts = []
    states_seq = []
    
    codons = random.choices(['A', 'C', 'G', 'U'], weights=np.ones(4), k = length[i])
    RNA_seq.extend(codons)
    states_seq.extend([1]*length[i])
    for j in range(length[i]):
      observed_counts.append(generate_nb(alpha_list_true, beta_list_true, 1, E[i]))

    RNA_data.append(RNA_seq)
    counts_data.append(observed_counts)
    states_data.append(states_seq)
    transition_index.append([0,0,0])
    
  return [RNA_data, counts_data, states_data, transition_index]
