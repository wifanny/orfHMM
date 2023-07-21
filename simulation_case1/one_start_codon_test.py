import sys
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time

from orfHMM_EM import EM_iter, forward_matrix, incomplete_log_likelihood
from orfHMM_Viterbi import viterbi_sequence
from SyncData import simulate_main, simulate_neither

r = 1
random.seed(r)
# set inputs
start_codon_list = [('AUG')]
transition_list = [[0.3, 0.6, 0.4], [0.3, 0.5, 0.2], [0.2, 0.4, 0.1]] # rho_u, rho, and delta

mean_list_true = np.array([10, 50, 45, 35, 25, 20, 15, 50, 40, 30, 10, 100, 90, 80, 60, 50, 30, 100, 95, 85, 10])
variance_list_true =  mean_list_true * 1.05
beta_list_true = 1/(variance_list_true/mean_list_true-1)
alpha_list_true = mean_list_true * beta_list_true

# generate 420 transcripts with translated regions
num1 = 420
E1 = [random.uniform(0.9, 1.1) for _ in range(num1)]
repeat_u = [random.randint(3, 20) for _ in range(num1)]
repeat = [random.randint(21, 60) for _ in range(num1)]
len_3UTR = [15]*num1
len_min_5UTR = [10]*num1
len_min_5UTR2 = [10]*num1

# generate 60 transcripts without translated regions
num2 = 60
length2 = [random.randint(100, 500) for _ in range(num2)]
E2 = [random.uniform(0.9, 1.1) for _ in range(num2)]

# generate datasets
part1 = simulate_main(num1, start_codon_list, transition_list, E1, 
              repeat_u, repeat, len_3UTR, len_min_5UTR, len_min_5UTR2, alpha_list_true, beta_list_true) 
part2 = simulate_neither(num2, length2, E2, alpha_list_true, beta_list_true)
E = E1+E2
RNA_data = part1[0] + part2[0]
observed_data = part1[1] + part2[1]
states_seq = part1[2] + part2[2]
transition_index = part1[3] + part2[3]

def count4(start_codon, RNA_data, states_seq, transition_index):
    count = 0
    for i in range(len(RNA_data)):
        if transition_index[i] == [1, 0, 1] or transition_index[i] == [1, 0, 0]:
            start = states_seq[i].index(2)
            RNA_seq = "".join(RNA_data[i][:(start+3)])
            count += RNA_seq.count(start_codon)
        if transition_index[i] == [0, 1, 0]:
            start = states_seq[i].index(12)
            RNA_seq = "".join(RNA_data[i][:(start+3)])
            count += RNA_seq.count(start_codon)
        if transition_index[i] == [0, 0, 0]:
            RNA_seq = "".join(RNA_data[i])
            count += RNA_seq.count(start_codon)
    return count


def count2(start_codon, RNA_data, states_seq, transition_index):
    count = 0
    for i in range(len(RNA_data)):
        if transition_index[i] == [1,0,1]:
            start = states_seq[i].index(10)
            stop = states_seq[i].index(12)
            RNA_seq = "".join(RNA_data[i][(start+1):(stop+3)])
            count += RNA_seq.count(start_codon)
        if transition_index[i] == [1,0,0]:
            start = states_seq[i].index(10)
            RNA_seq = "".join(RNA_data[i][(start+1):])
            count += RNA_seq.count(start_codon)
        else:
            count += 0
    return count
# total amount of AUG in 4 types of transcripts
count_AUG1 = count4('AUG', RNA_data, states_seq, transition_index)

# total amount of AUG in 2 types of transcripts
count_AUG2 = count2('AUG', RNA_data, states_seq, transition_index)


# count start codons for 4 types of transcripts
# compute real rho_u
count11 = 0 #AUG
count12 = 0 #CUG
count13 = 0 #GUG
for i in range(len(states_seq)):
    if transition_index[i] == [1,0,1] or transition_index[i] == [1,0,0]:
        index_2 = states_seq[i].index(2)
        start = RNA_data[i][index_2]+RNA_data[i][index_2+1]+RNA_data[i][index_2+2]
        if start == 'AUG':
            count11 += 1 
        if start == 'CUG':
            count12 += 1 
        if start == 'GUG':
            count13 += 1
            
# compute real rho
count21 = 0 #AUG
count22 = 0 #CUG
count23 = 0 #GUG
for i in range(len(states_seq)):
    if transition_index[i] == [0,1,0]:
        index_12 = states_seq[i].index(12)
        start = RNA_data[i][index_12]+RNA_data[i][index_12+1]+RNA_data[i][index_12+2]
        if start == 'AUG':
            count21 += 1 
        if start == 'CUG':
            count22 += 1 
        if start == 'GUG':
            count23 += 1

# compute real delta
# count start codons for 2 types of transcripts
count31 = 0 #AUG
count32 = 0 #CUG
count33 = 0 #GUG
for i in range(len(states_seq)):
    if transition_index[i] == [1,0,1]:
        index_12 = states_seq[i].index(12)
        start = RNA_data[i][index_12]+RNA_data[i][index_12+1]+RNA_data[i][index_12+2]
        if start == 'AUG':
            count31 += 1 
        if start == 'CUG':
            count32 += 1 
        if start == 'GUG':
            count33 += 1   
trans_real = {'AUG': [count11/count_AUG1, count21/count_AUG1, count31/count_AUG2]}


# set initial values
stop_codon_list = ['UAA', 'UGA', 'UAG']
start_codon_list = ['AUG']

loglik_max = []
loglik_curve = []
trans = []
alpha = []
beta = []
diff_count = []
diff_sites = []
for init in range(4):
    # true: rho_u = 0.3, rho = 0.6, delta = 0.4
    a = random.uniform(0.13,0.23)
    b = random.uniform(0.2,0.42)
    c = random.uniform(0.13,0.23)
    trans_prob_list = [a, b, c]
    trans_init = {}
    trans_init['AUG'] = trans_prob_list

    # true: 200. 1000.  900.  700.  500.  400.  300. 1000.  800.  600.  200. 2000. 1800. 1600. 1200. 1000.  600. 2000. 1900. 1700.  200.
    alpha_list_init = []
    for i in range(len(alpha_list_true)):
        alpha_list_init.append(random.uniform(alpha_list_true[i]*(1-0.1),alpha_list_true[i]*(1+0.1)))
    # true: 20
    beta_list_init = []
    for i in range(len(beta_list_true)):
        beta_list_init.append(random.uniform(beta_list_true[i]*(1-0.1),beta_list_true[i]*(1+0.1)))


    # parameter estimation

    # update both alpha and beta
    # Here's the EM algorithm
    temp = EM_iter(RNA_data, observed_data, E, trans_init, alpha_list_init, 
                    beta_list_init, 1e-10, 100, False, stop_codon_list, True)
    trans_est= temp[0]
    alpha_list_est = temp[1]
    beta_list_est = temp[2]
    incomplete_loglik = temp[3]
    # record 
    trans.append(trans_est)
    alpha.append(alpha_list_est)
    beta.append(beta_list_est)
    loglik_max.append(incomplete_loglik[-1])
    loglik_curve.append(incomplete_loglik)

    # infer the translated regions after mplementing EM algorithm
    viterbi_output = viterbi_sequence(RNA_data, observed_data, 
                                          alpha_list_est, beta_list_est, 
                                    E, trans_est, stop_codon_list, True) 
    
    # count difference between inferred ones and true ones
    unmatch_count= 0
    for i in range(len(states_seq)):
        for j in range(len(states_seq[i])):
            if viterbi_output[i][j] != states_seq[i][j]:
                unmatch_count += 1

    diff_counts =[unmatch_count]
    diff_count.append(diff_counts)

    # find the identified translated regions
    if unmatch_count != 0:
        true_mat = np.zeros((len(RNA_data),4))
        infer_mat = np.zeros((len(RNA_data),4))
        for i in range(len(states_seq)):
            if (2 in states_seq[i]):
                true_mat[i,0] = states_seq[i].index(2)
            if (8 in states_seq[i]):
                true_mat[i,1] = states_seq[i].index(8)
            if (12 in states_seq[i]):
                true_mat[i,2] = states_seq[i].index(12)
            if (18 in states_seq[i]):
                true_mat[i,3] = states_seq[i].index(18)

            if (2 in viterbi_output[i]):
                infer_mat[i,0] = viterbi_output[i].index(2)
            if (8 in viterbi_output[i]):
                infer_mat[i,1] = viterbi_output[i].index(8)
            if (12 in viterbi_output[i]):
                infer_mat[i,2] = viterbi_output[i].index(12)
            if (18 in viterbi_output[i]):
                infer_mat[i,3] = viterbi_output[i].index(18)

        diff_site = pd.DataFrame(np.hstack((true_mat,infer_mat)))
    else:
        diff_site = pd.DataFrame(columns=[0,1,2,3,4,5,6,7,8], index=[0])
    
    diff_sites.append(diff_site) 


max_index = loglik_max.index(max(loglik_max))
# save as files
# alpha and beta names
alpha_name_list = []
for i in range(1, 22):
    alpha_name_list.append(str(i))

beta_name_list = []
for i in range(1, 22):
    beta_name_list.append(str(i))

# construct final alpha dataframe
alpha_df_est = pd.DataFrame([alpha[max_index]], columns=alpha_name_list)

# construct final beta dataframe
beta_df_est = pd.DataFrame([beta[max_index]], columns=beta_name_list)

# construct trans probs dataframe
for element in trans[max_index].keys():
    rou_u1 = [trans[max_index][element][0]]
    rou1 = [trans[max_index][element][1]]
    delta1 = [trans[max_index][element][2]]
trans_df_est = pd.DataFrame({"rou_u":rou_u1, "rou":rou1,"delta": delta1}, 
                                         columns=['rou_u', 'rou', 'delta'])

for element in trans_real.keys():
    rou_u1 = [trans_real[element][0]]
    rou1 = [trans_real[element][1]]
    delta1 = [trans_real[element][2]]
trans_df_real = pd.DataFrame({"rou_u":rou_u1, "rou":rou1,"delta": delta1}, 
                                         columns=['rou_u', 'rou', 'delta'])

                                      
# construct diff counts dataframe
diff_count = pd.DataFrame(diff_count[max_index])
diff_site = pd.DataFrame(diff_sites[max_index])

def list_to_array (x):
    dff = pd.concat([pd.DataFrame({'{}'.format(index):labels}) for index,labels in enumerate(x)],axis=1)
    return dff.fillna(0).values.T.astype(float)

loglik_curve = list_to_array(loglik_curve)
loglik_curve = pd.concat([pd.DataFrame({'{}'.format(index):labels}) for index,labels in enumerate(loglik_curve)],axis=1)

# compute the maximum incomplete log-likelihood from true emission parameters
forward_est = forward_matrix(RNA_data, observed_data, alpha_list_true, beta_list_true, E, trans_est, stop_codon_list, True)
max_incomplete_loglik = incomplete_log_likelihood(forward_est, observed_data, E)
max_incomplete = {r: max_incomplete_loglik}
max_incomplete_df=pd.DataFrame.from_dict(max_incomplete, orient='index')
   
    

