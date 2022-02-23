import numpy as np
import math
import pandas as pd
import json
from math import log10, floor


""" The so-called future is no more than a result from the decision that we made today. 


                                                                                                    --- Mahabharata """

"""Helper Functions"""

def generate_independent_sequence(n_sample = 1000):
    stim_set = [(0,),(1,1,1),(2,)]
    sequence = []
    states = [[0], [1,1,1], [2]]
        # sample from an arbitrary distribution according to stimulus states and cumulative distribution function
    prob = [0.40,0.30,0.30]
    cdf = [0.0]
    for s in range(0, len(states)):
        cdf.append(cdf[s] + prob[s])
    for n in range(0, n_sample):
        k = np.random.rand()
        for i in range(1,len(states)+1):
            if (k>=cdf[i-1]):
                if (k < cdf[i]):
                    this_sample =  states[i-1]
                    for item in this_sample:
                        sequence.append(item)
    return sequence,stim_set

def generate_markov_chain_sequence(n_sample = 1000):
    stim_set = [(1,2),(3,4)]
    transition = {(1,2): {(1,2): 0.7 , (3,4): 0.3}, (3,4):{(1,2): 0.2, (3,4): 0.8}}
    marginals = {(1,2):0.5,(3,4):0.5}
    sequence = []
    i = 0
    s_first = sample_from_distribution(list(marginals.keys()), list(marginals.values()))
    s_first = [3,4]
    sequence = sequence + s_first
    s_last = s_first
    while i <= n_sample:
        i = i + 1
        new_sample = sample(transition, marginals, s_last)
        s_last = new_sample
        sequence = sequence + new_sample
    
    return sequence, stim_set



def generate_sequence_with_chunks(n_sample = 1000):
    stim_set = [(1,1,2),(0,1),(2,2),(3,4)]
    transition = {(1,1,2): {(3,4): 0.8, (2,2): 0.2}, 
                  (0,1): {(3,4): 0.2, (2,2): 0.8},
                  (2,2):{(1,1,2):0.4, (0,1):0.5,(2,2):0.1},
                  (3,4):{(1,1,2):0.4, (0,1):0.6,(3,4):0.1}}
    marginals = {(1,1,2):0.2,(0,1):0.2,(3,4):0.4,(2,2):0.2}
    sequence = []
    i = 0
    s_first = sample_from_distribution(list(marginals.keys()), list(marginals.values()))
    s_first = [3,4]
    sequence = sequence + s_first
    s_last = s_first
    while i <= n_sample:
        i = i + 1
        new_sample = sample(transition, marginals, s_last)
        s_last = new_sample
        sequence = sequence + new_sample
    return sequence,stim_set

def generate_probabilistic_deterministic_sequence(n_sample = 1000):
    stim_set = [(1,),(2,),(3,),(4,)]
    transition = {(1,): {(1,): 0.00, (2,): 0.8 , (3,): 0.20,(4,): 0.0}, 
                  (2,): {(1,): 0.15 ,(2,): 0.05 , (3,): 0.6,(4,): 0.2},
                  (3,): {(1,): 0.0 ,(2,): 1.0 , (3,): 0.0,(4,): 0.0},
                  (4,): {(1,): 0.85 ,(2,): 0.05 , (3,): 0.05,(4,): 0.05}}
    marginals = {(1,):0.25,(2,):0.25,(3,):0.25, (4,):0.25}
    sequence = []
    i = 0
    s_first = list(sample_from_distribution(list(marginals.keys()), list(marginals.values())))
    sequence = sequence + s_first
    s_last = s_first
    while i <= n_sample:
        i = i + 1
        new_sample = sample(transition, marginals, s_last)
#         print('new sample is', new_sample)
        s_last = new_sample
        sequence = sequence + new_sample
    
    return sequence,stim_set



def generate_random_chunk_sequence(n_sample= 1000, pauses = False):
    # randomly compose chunks with random size, pauses refer to a break time where no instruction is given. 
    def compose_one_chunk(size, instr_set):
        chunk = []
        for i in range(0,size):
            chunk.append(np.random.choice(instr_set))
        return chunk
    n_chunks = np.random.choice([4,5,6]) # get the number of chunks
    size_of_chunk = [2,3,4,5,6]
    single_instr = [1,2,3,4]
    # for each chunk, get the size of it. 
    chunks = []
    for i in range(n_chunks): 
        thischunksize = np.random.choice(size_of_chunk)
        chunks.append(compose_one_chunk(thischunksize, single_instr))# a chunk with random instruction specified by the size
    sequence = []
    for i in range(n_sample):
        sequence+=np.random.choice(chunks)
        if pauses: sequence.append(np.random.choice([0,]))

    return sequence, chunks



def generate_ABCD_sequences(n_sample = 1100,H = 0.8, M = 0.4):

    stim_set = [(1,),(2,),(3,),(4,)]
    transition = {(1,): {(1,): (1-H)/3., (2,): H , (3,):(1-H)/3. ,(4,): (1-H)/3.}, 
                  (2,): {(1,): (1-M)/3. ,(2,): (1-M)/3., (3,): M ,(4,): (1-M)/3.},
                  (3,): {(1,): (1-H)/3. ,(2,): (1-H)/3. , (3,): (1-H)/3.,(4,): H},
                  (4,): {(1,): M ,(2,): (1-M)/3., (3,): (1-M)/3.,(4,): (1-M)/3.}}
    
    marginals = {(1,):0.25,(2,):0.25,(3,):0.25, (4,):0.25}
    sequence = []
    i = 1
    s_first,_ = list(sample_from_distribution(list(marginals.keys()), list(marginals.values())))
    sequence = sequence + s_first
    s_last = s_first
    while i < n_sample:
        i = i + 1
        new_sample,_ = sample(transition, marginals, s_last)
#         print('new sample is', new_sample)
        s_last = new_sample
        sequence = sequence + new_sample
    return sequence,stim_set


def generate_independent_instr(n_sample = 1000):
    sequence = []
    instr = [1,2,3,4]
    for i in range(0,n_sample): 
        sequence.append(np.random.choice(instr))
    return sequence, instr




#Generate ASRT data
def generate_ASRT_data(stim_length = 500):
    stim_set = [1, 2, 3, 4]
    stim_cont = []
    i=0
    while i < stim_length:
        if i%2:
            current_word = np.random.choice(stim_set)
        else:
            current_word = stim_set[int((i/2)%4)]
        stim_cont = stim_cont + [current_word]
        i+=1
        
    return stim_cont,stim_set

def get_empirical_probability(chunk1,chunk2,generated_sequence):
    '''Get the estimated empirical probability of P(chunk2|chunk1),the probability of chunk2 followed by chunk1, 
    in the generated sequence
    In the case when chunk1 never occurs, output the probability of 0'''
    chunk1 = list(chunk1)
    chunk2 = list(chunk2)
    c_len1 = len(list(chunk1))
    c_len2 = len(list(chunk2))
    chunk1_count = 0
    chunk1_chunk2 = 0
    # the transition probability from chunk1 to chunk2
    # get P(chunk2|chunk1)
    not_over = True
    i = 0
    while not_over:
        candidate = generated_sequence[i:i+c_len1]
        if candidate == chunk1:
            chunk1_count +=1
            i = i + c_len1
            candidate2 = generated_sequence[i:i+c_len2]
            if candidate2 == chunk2:
                chunk1_chunk2+=1
                i = i + c_len2
        else:
            i = i + 1
        if i >= len(generated_sequence):
            not_over = False
    if chunk1_count>0:
        return chunk1_chunk2/chunk1_count
    else:
        return 0.0

    
# print((sequence))

def measure_total_entropy(marginals, transition_matrix):
    marginal_entropy = 0
    for m in list(marginals.values()):
        if m >0: 
            marginal_entropy = marginal_entropy + -math.log2(m)
    transition_entropy = 0
    for key in list(transition_matrix.keys()):
        probs = list(transition_matrix[key].values())
        probs = [item/sum(probs) for item in probs]
        for p in probs:
            if p>0.0:
                transition_entropy = transition_entropy + -math.log2(p)
    return marginal_entropy, transition_entropy




     

def get_transition_p(transition_matrix,symbol_table):
    T_P = np.zeros([4,4])
    # get the frequency of occurrance for ith item
    f_i = np.zeros([4,1])
    for i in range(1,5):
        for key in list(symbol_table.keys()):
            f_i[i-1]+=list(key).count(i)*symbol_table[key]
    f_ij = np.zeros([4,4])
    for i in range(1,5):
        for j in range(1,5):
            # iterate for each chunk
            for key in list(symbol_table.keys()):
                ij = "".join(map(str,[i,j]))
                f_ij[i-1,j-1]+="".join(map(str,list(key))).count(ij)*symbol_table[key] 
            # iterate for each transition
            for key in list(transition_matrix.keys()):
                # last item of key:
                if i == list(key)[-1]:
                    if transition_matrix[key]!={}:
                        for key2 in list(transition_matrix[key].keys()):
                            if j == list(key2)[0]:
                                f_ij[i-1,j-1]+=transition_matrix[key][key2]
    return f_ij/f_i
        

    
def get_transition_p_1(transition_matrix,symbol_table, previous_observation, future_observation, predicted_observation):
    # iterate until the very end of the future observation
    # if there are transition matrix elements that can explain, then use the transition elements.
    # Otherwise, use the marginal probability elements. 
    transition = []
#     print(previous_observation,future_observation, predicted_observation)
    if tuple(previous_observation) in list(transition_matrix.keys()):
        transition_entry = transition_matrix[tuple(previous_observation)]
        for i in range(0,len(future_observation)):
            T_P = np.zeros([4,1])  
            elements_to_explain = future_observation[0:i+1]
            for j in range(0,4): 
                elements_to_explain[-1] = j+1# iterate over all possibilities of 1,2,3,4
                candidate_keys = [key for key in transition_entry.keys() if list(key)[0:i+1] == elements_to_explain]
                if candidate_keys!=[]:
                    T_P[j] = sum([transition_entry[key] for key in candidate_keys])
                else:
                    candidate_keys = [key for key in symbol_table.keys() if list(key)[0:i+1]== elements_to_explain]
                    T_P[j] = sum([symbol_table[key] for key in candidate_keys])     
            T_P = T_P/sum(T_P)
            transition = transition +[T_P]
    else: # previous observation not in the list of keys in transition matrix
        for i in range(0,len(future_observation)):
            T_P = np.zeros([4,1]) 
            elements_to_explain = future_observation[0:i+1]
            for j in range(0,4):             
                elements_to_explain[-1] = j+1
                candidate_keys = [ key for key in symbol_table.keys() if list(key)[0:i+1] == elements_to_explain]
                if candidate_keys!=[]:
                    T_P[j] = sum([symbol_table[key] for key in candidate_keys])
                else: # there are no marginals to explain this observation either. 
                    T_P[j] = 1       
            T_P = T_P/sum(T_P)
            transition = transition +[T_P]
    return transition
        
        
def get_candidate_key(this_dictionary,elements_to_explain):
    candidate_keys = []
    for key in this_dictionary.keys():
        if list(key)[0:i] == elements_to_explain:
            candidate_keys.append(key)
    return candidate_keys

def reproduce_sequence(marginals, transition_matrix,n):
    # reproduce sequence based on the learned generative model. 
    produced_sequence = []
    s_last = [3]
    for i in range(0,n):
        item = sample(transition_matrix, marginals, s_last)
        produced_sequence += s_last
        s_last = item
    # print(produced_sequence)
    return produced_sequence

# # find the state with the lowest entropy
# in other words, I want to maximimze my certainty about what is going to happen next. 
def pick_observation(o): 
    observation_candidates = []
    for m in symbol_table.keys():
        length_m = len(list(m))
        observation = sequence[o-length_m:o]
        print('s_last ', observation, ', m ', m)
        if observation == list(m):
            observation_candidates.append(m)
    Entropy = []
    print('s_last candidates are: ', observation_candidates)
    if observation_candidates == []:
        min_entropy_observation = sequence[o]
    else:
        for candidate in observation_candidates:
            Entropy.append(measure_entropy(transition_matrix, candidate))
        print('choice entropy is: ', Entropy)
        min_entropy_choice = Entropy.index(min(Entropy))
        min_entropy_observation = list(symbol_table.keys())[min_entropy_choice]
    return min_entropy_observation

    
def measure_entropy(transition_matrix, state):
    E = 0
    transition_prob = {}
    if state in transition_matrix.keys():
        if list(transition_matrix[state].keys()) == []:
            E = 0
        else:
            for key in transition_matrix[state].keys():
                transition_prob[key] = transition_matrix[state][key]/sum(transition_matrix[state].values())
                if transition_prob[key]>=0:
                    E = E - transition_prob[key]*np.log(transition_prob[key])
    else:
        E = 0
    return E
        

def sample_from_distribution(states,prob):
    """
    states: a list 
    prob: another list that contains the probability"""
    prob =  [k/sum(prob) for k in prob]
    cdf = [0.0]
    for s in range(0, len(states)):
        cdf.append(cdf[s] + prob[s])
    k = np.random.rand()
    for i in range(1,len(states)+1):
        if (k>=cdf[i-1]):
            if (k < cdf[i]):
                return list(states[i-1]),prob[i-1]
            
def sample(transition, marg, s_l):
    """When it returns [], it means there is no prediction,
        otherwise, returns the predicted sequence of certain length as a list
        s_last: a tuple, of last stimuli, as the key to look up in the transition probability dictionary"""
    s_l = tuple(s_l)
    if transition == {}:
        return [],0
    elif s_l not in list(transition.keys()): 
        # transition matrix empty, sample from marginals
        # print('no key in transition matrix')

        states = list(marg.keys())
        prob = []
        for s in range(0, len(states)):
            prob.append(marg[states[s]])
        prob =  [k/sum(prob) for k in prob]
        return sample_from_distribution(states,prob)        
    elif list(transition[s_l].keys()) == []: 
        # transition matrix have seen this s_last before, but don't know what is next, 
        # still sample from that marginal probability
#         print('no transition in transition matrix')
        states = list(marg.keys())
        prob = []
        for s in range(0, len(states)):
            prob.append(marg[states[s]])
        prob =  [k/sum(prob) for k in prob]
        return sample_from_distribution(states,prob)
    else:
        # print('indeed sampling from distribution')
        states = list(transition[s_l].keys())
        # sample from an arbitrary distribution according to stimulus states and cumulative distribution function
        prob = []
        for s in range(0, len(states)):
            prob.append(transition[s_l][states[s]])
        prob =  [k/sum(prob) for k in prob]
        
        return sample_from_distribution(states,prob)



def get_transition_p_1(transition_matrix,symbol_table, previous_observation, future_observation, predicted_observation):
    # iterate until the very end of the future observation
    # if there are transition matrix elements that can explain, then use the transition elements.
    # Otherwise, use the marginal probability elements. 
    transition = []
#     print(previous_observation,future_observation, predicted_observation)
    if tuple(previous_observation) in list(transition_matrix.keys()):
        transition_entry = transition_matrix[tuple(previous_observation)]
        for i in range(0,len(future_observation)):
            T_P = np.zeros([4,1])  
            elements_to_explain = future_observation[0:i+1]
            for j in range(0,4): 
                elements_to_explain[-1] = j+1# iterate over all possibilities of 1,2,3,4
                candidate_keys = [key for key in transition_entry.keys() if list(key)[0:i+1] == elements_to_explain]
                if candidate_keys!=[]:
                    T_P[j] = sum([transition_entry[key] for key in candidate_keys])
                else:
                    candidate_keys = [key for key in symbol_table.keys() if list(key)[0:i+1]== elements_to_explain]
                    T_P[j] = sum([symbol_table[key] for key in candidate_keys])     
            T_P = T_P/sum(T_P)
            transition = transition +[T_P]
    else: # previous observation not in the list of keys in transition matrix
        for i in range(0,len(future_observation)):
            T_P = np.zeros([4,1]) 
            elements_to_explain = future_observation[0:i+1]
            for j in range(0,4):             
                elements_to_explain[-1] = j+1
                candidate_keys = [ key for key in symbol_table.keys() if list(key)[0:i+1] == elements_to_explain]
                if candidate_keys!=[]:
                    T_P[j] = sum([symbol_table[key] for key in candidate_keys])
                else: # there are no marginals to explain this observation either. 
                    T_P[j] = 1       
            T_P = T_P/sum(T_P)
            transition = transition +[T_P]
    return transition
        

def learning(s, s_last,symbol_table, marginals,transition_matrix,RT_w, RT_b, w = 0.5, theta = 0.95, deletion_threshold = 0.5, Reward_structure=None, Punish_structure = None):

    '''
    update the marginal probability, the symbol table, based on whether this_observation is a new instance
    symbol_table: table with counting frequency of each observed symbol
    this_observation: tuple on this observation
    # reward_structure: dictionary with elements correspond to rewards
    # punishment_structure: dictionary with elements correspon to punishment on misses. 
    '''

    # print('s is ', s, 's_last is ', s_last)
    # print('w' , w)
    chunked = False
    s = tuple(s)
    s_last = tuple(s_last)
    cat =  list(s_last)+list(s)


    if s not in list(symbol_table.keys()):
        symbol_table[s] = 1.0
    else:
        symbol_table[s] = symbol_table[s] + 1.0

    # transition matrix contains the last two time points
    # also, how often would you be able to update?
    # if very frequent, up to a threshold, then combine the two into one pattern. 
    if s_last in list(transition_matrix.keys()):
        if s in list(transition_matrix[s_last].keys()):
            transition_matrix[s_last][s] = transition_matrix[s_last][s] + 1
        else: 
            transition_matrix[s_last][s] = 1 

    else:
        transition_matrix[s_last] = {}
        transition_matrix[s_last][s] = 1

    satisfy_criteria = False # the criteria for chunking 

    if s_last  in list(symbol_table.keys()):
        if s in list(transition_matrix[s_last].keys()):
            # print(symbol_table[s_last],transition_matrix[s_last][s])
            if symbol_table[s_last]>1.0:
                if transition_matrix[s_last][s] > 1.0: # strangely, this number affects the chunking behavior a lot. 
                    satisfy_criteria = True

    # print(satisfy_criteria）
    for key in list(symbol_table.keys()):
        marginals[key] = symbol_table[key]/sum(symbol_table.values())

    if satisfy_criteria: # decide if one should chunk or not
        sum_transition = 0
        for key in list(transition_matrix[s_last].keys()):
            sum_transition += transition_matrix[s_last][key]
        Expected_reward_cat = 0
        P_s_last = marginals[s_last]
        P_s_giv_s_last = transition_matrix[s_last][s]/sum_transition
        symbol_table_copy = symbol_table.copy()
        transition_matrix_copy = transition_matrix.copy()
        marginals_copy = marginals.copy()
        Expected_reward_cat = rational_chunking(marginals, transition_matrix,s_last,s,w, Reward_structure,Punishment_structure)
        if Expected_reward_cat>0:
            chunked = True
            symbol_table,transition_matrix, marginals = chunking_reorganization(symbol_table_copy,transition_matrix_copy, marginals_copy,s_last,s,cat, P_s_last,P_s_giv_s_last)
        else: chunked = False

    # if Expected_reward_cat > 0: symbol_table,transition_matrix, marginals = chunking_reorganization(symbol_table_copy,transition_matrix_copy, marginals_copy,s_last,s,cat, P_s_last,P_s_giv_s_last) # rational chunking is satisfied. 

    transition_matrix,marginals, symbol_table = forgetting(symbol_table,deletion_threshold,marginals,transition_matrix,theta)

    # print('symbol table here is: ', symbol_table)
        ## renormalize the marginals 

    total_value = sum(symbol_table.values())
    for key in list(symbol_table.keys()):
        marginals[key] = symbol_table[key]/total_value

    return transition_matrix,symbol_table, marginals, chunked


def step_minimization_learning(s, s_last,symbol_table, marginals,transition_matrix,RT_w, RT_b,old_L, theta = 0.95, deletion_threshold = 0.5, AFAP = False, AAAP = False, Reward_structure=None, Punish_structure = None):

    '''
    learning by minimizing cost function
    '''
    # print('w' , w)
    chunked = False
    s = tuple(s)
    s_last = tuple(s_last)
    cat =  list(s_last)+list(s)
    new_L = old_L


    if s not in list(symbol_table.keys()):
        symbol_table[s] = 1.0
    else:
        symbol_table[s] = symbol_table[s] + 1.0

    # transition matrix contains the last two time points
    # also, how often would you be able to update?
    # if very frequent, up to a threshold, then combine the two into one pattern. 
    if s_last in list(transition_matrix.keys()):
        if s in list(transition_matrix[s_last].keys()):
            transition_matrix[s_last][s] = transition_matrix[s_last][s] + 1
        else: 
            transition_matrix[s_last][s] = 1 

    else:
        transition_matrix[s_last] = {}
        transition_matrix[s_last][s] = 1

    # print(satisfy_criteria）
    for key in list(symbol_table.keys()):
        marginals[key] = symbol_table[key]/sum(symbol_table.values())


    if AFAP: 
        L_speed_original = evaluate_L(marginals, transition_matrix, RT_w, RT_b, AAAP = False, AFAP = True)
        L_update = L_speed_original
    elif AAAP: 
        L_error_original = evaluate_L(marginals, transition_matrix, RT_w, RT_b, AAAP = True, AFAP = False)
        L_update = L_error_original

    else: # reward assymmetry
        L_Expected_reward_original = evaluate_assymetry_reward(marginals, transition_matrix, Reward_structure,Punishment_structure)
        L_update = L_Expected_reward_original

    new_L = L_update

    satisfy_criteria = False # the criteria for chunking 
    # in order to satisfy criteria of chunking, loss estimation needs to be stationary. 
    satisfy_criteria1 = False # the criteria for chunking 
    satisfy_criteria2 = False # the criteria for chunking 

    if s_last  in list(symbol_table.keys()):
        if s in list(transition_matrix[s_last].keys()):
            # print(symbol_table[s_last],transition_matrix[s_last][s])
            if symbol_table[s_last]>1.0:
                if transition_matrix[s_last][s] > 1.0: # strangely, this number affects the chunking behavior a lot. 
                    satisfy_criteria1 = True

    if np.abs(L_update - old_L) < 0.2: # make sure the update is stable. 
        satisfy_criteria2 = True 

    satisfy_criteria = satisfy_criteria1 and satisfy_criteria2
    print('satisfy criteria', satisfy_criteria)

    if satisfy_criteria: # decide if one should chunk or not
        sum_transition = 0
        for key in list(transition_matrix[s_last].keys()):
            sum_transition += transition_matrix[s_last][key]
        Expected_reward_cat = 0
        P_s_last = marginals[s_last]
        P_s_giv_s_last = transition_matrix[s_last][s]/sum_transition
        symbol_table_copy = symbol_table.copy()
        transition_matrix_copy = transition_matrix.copy()
        marginals_copy = marginals.copy()

        if AFAP: 
            symbol_table_copy,transition_matrix_copy, marginals_copy = chunking_reorganization(symbol_table_copy,transition_matrix_copy, marginals_copy,s_last,s,cat, P_s_last,P_s_giv_s_last)
            L_speed_chunk = evaluate_L(marginals_copy, transition_matrix_copy, RT_w, RT_b, AAAP = False, AFAP = True)
            diff_L_speed = L_speed_chunk - L_speed_original
            chunked = False
            if diff_L_speed<0:# speed is decreasing
                chunked = True
                symbol_table,transition_matrix, marginals = symbol_table_copy,transition_matrix_copy, marginals_copy
                new_L = L_speed_chunk
            else:
                new_L = L_speed_original

        elif AAAP: 
            symbol_table_copy,transition_matrix_copy, marginals_copy = chunking_reorganization(symbol_table_copy,transition_matrix_copy, marginals_copy,s_last,s,cat, P_s_last,P_s_giv_s_last)
            L_error_chunk = evaluate_L(marginals_copy, transition_matrix_copy, RT_w, RT_b, AAAP = True, AFAP = False)
            diff_L_error= L_error_chunk - L_error_original 
            chunked = False
            if diff_L_error<0:# error rate is decreasing
                symbol_table,transition_matrix, marginals = symbol_table_copy,transition_matrix_copy, marginals_copy
                chunked = True
                new_L = L_error_chunk
            else:
                new_L = L_error_original

        else: # reward assymmetry
            symbol_table_copy,transition_matrix_copy, marginals_copy = chunking_reorganization(symbol_table_copy,transition_matrix_copy, marginals_copy,s_last,s,cat, P_s_last,P_s_giv_s_last)
            L_Expected_reward_chunk = evaluate_assymetry_reward(marginals_copy, transition_matrix_copy, Reward_structure,Punishment_structure)
            diff_L_reward = L_Expected_reward_chunk - L_Expected_reward_original # chunk if expected reward increases
            if diff_L_reward>0:
                chunked = True
                symbol_table,transition_matrix, marginals = chunking_reorganization(symbol_table_copy,transition_matrix_copy, marginals_copy,s_last,s,cat, P_s_last,P_s_giv_s_last)
                new_L = L_Expected_reward_chunk
            else:
                new_L = L_Expected_reward_original

    # if Expected_reward_cat > 0: symbol_table,transition_matrix, marginals = chunking_reorganization(symbol_table_copy,transition_matrix_copy, marginals_copy,s_last,s,cat, P_s_last,P_s_giv_s_last) # rational chunking is satisfied. 

    transition_matrix,marginals, symbol_table = forgetting(symbol_table,deletion_threshold,marginals,transition_matrix,theta)

    # print('symbol table here is: ', symbol_table)
        ## renormalize the marginals 

    total_value = sum(symbol_table.values())
    for key in list(symbol_table.keys()):
        marginals[key] = symbol_table[key]/total_value

    return transition_matrix,symbol_table, marginals, chunked, new_L



## TODO: evaluate assymetric reward:  
def evaluate_assymetry_reward(marginals, transition_matrix, Reward_structure,Punishment_structure):
    #P(correct)* reward + (1-P(correct))*punishment
    # given a bag of chunks and transitions, evaluate the expected reward and punishment structure
    L = 0

    for item in marginals.keys():
        P_prev = marginals[item]
        if item in transition_matrix.keys():
            sum_transition_item = 0
            for next in transition_matrix[item].keys():
                sum_transition_item+=transition_matrix[item][next]
            for next in transition_matrix[item].keys():
                P_next_giv_prev = transition_matrix[item][next]/sum_transition_item

                if Reward_structure!=None:
                    R_chunk = 0
                    for i in list(next):
                        R_chunk = R_chunk + Reward_structure[tuple([i])]
                    L+=P_prev*P_next_giv_prev*R_chunk
                else:
                    L+=P_prev*P_next_giv_prev*len(next)

                # print("expected reward positive part: ", Expected_reward_cat )
                if Punish_structure!=None:
                    P_chunk = 0
                    for i in list(key):
                        P_chunk = P_chunk + Punish_structure[tuple([i])]
                    L+=P_prev*(1-P_next_giv_prev)*P_chunk
                else: 
                    L+=P_prev*(1-P_next_giv_prev)*(-len(next))


    return L


def evaluate_L(marginals, transition_matrix, RT_w, RT_b, AAAP = False, AFAP = False, prev = None):
    if AAAP: 
        L = 0
        if list(marginals.keys())!=[]:
            for item in marginals.keys():
                P_prev = marginals[item]
                if item in transition_matrix.keys():
                    sum_transition_item = 0
                    if list(transition_matrix[item].keys())!=[]:
                        for next in transition_matrix[item].keys():
                            sum_transition_item+=transition_matrix[item][next]
                        for next in transition_matrix[item].keys():
                            L+=P_prev*(1-(transition_matrix[item][next]/sum_transition_item)**2)#*len(next) why would L be bigger than 1? 
        else:
            L = 1000

    if AFAP: 
        L = 0
        if list(marginals.keys())!=[]:
            for item in marginals.keys():
                P_prev = marginals[item]
                if item in transition_matrix.keys():
                    sum_transition_item = 0
                    if list(transition_matrix[item].keys())!=[]:
                        for next in transition_matrix[item].keys():
                            sum_transition_item+=transition_matrix[item][next]
                        for next in transition_matrix[item].keys():
                            P_obs_giv_prev = transition_matrix[item][next]/sum_transition_item
                            if P_obs_giv_prev>0:
                                L+=P_prev*P_obs_giv_prev*(-np.log(P_obs_giv_prev) + RT_w*(len(next)-1)+ RT_b)/len(next)
        else:
            L = 1000
    print("L is: ", L )
    return L




# TODO: Evaluate the hierarchical generative model on the ground truth reproduced sequence. 
def evaluate_KL_compared_to_ground_truth(reproduced_sequence, generative_marginals):
    """compute conditional KL divergence between the reproduced sequence and the groundtruth"""
    """generative_marginal: marginals used in generating ground truth """
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """     
    ground_truth_set_of_chunks = set(generative_marginals.keys())
    partitioned_seq_ground_truth_chunk = partition_seq(reproduced_sequence,ground_truth_set_of_chunks)
    learned_M,_ = get_M_T_from_partitioned_sequence(partitioned_seq_ground_truth_chunk)
    # compare the learned M with the generative marginals
    # Iterate over dictionary keys, and add key values to the np.array to be compared
    # based on the assumption that the generative marginals should have the same key as the probability ground truth.
    probability_learned = []
    probability_ground_truth = []
    for key in list(learned_M.keys()):
        probability_learned.append(learned_M[key])
        probability_ground_truth.append(generative_marginals[key])
    probability_learned= np.array(probability_learned)
    probability_ground_truth = np.array(probability_ground_truth)
    eps = 0.000000001
    EPS = np.ones(probability_ground_truth.shape)*eps
    # return divergence
    v_M1 = probability_ground_truth
    v_M1 = EPS + v_M1
    v_M2 = probability_learned
    v_M2 = EPS + v_M2
    p_log_p_div_q = np.multiply(v_M1,np.log(v_M1/v_M2)) # element wise multiplication
    div = np.sum(np.matmul(v_M1.transpose(),p_log_p_div_q))
    return div


def chunking_reorganization(symbol_table_copy,transition_matrix_copy, marginals_copy,s_last,s,cat, P_s_last,P_s_giv_s_last):
    '''
    Reorganize marginal and transitional probability matrix

    '''
    """Model hasn't seen this chunk before:""" 
    if (tuple(cat) not in list(symbol_table_copy.keys())) & (s_last != []) & (tuple(s) in list(symbol_table_copy.keys())):
        ## estimate the marginal probability of cat from P(s_last)P(s|s_last)
        symbol_table_copy[tuple(cat)] =  P_s_giv_s_last*P_s_last*sum(symbol_table_copy.values())
        if transition_matrix_copy != {}:                       
            if tuple(s) in list(transition_matrix_copy.keys()):
                transition_matrix_copy[tuple(cat)] = transition_matrix_copy[tuple(s)].copy()
            for key in list(transition_matrix_copy.keys()):       
                if tuple(s_last) in list(transition_matrix_copy[key].keys()):
                    transition_matrix_copy[key][tuple(cat)] = 1  
                    transition_matrix_copy[key][tuple(s_last)] =  transition_matrix_copy[key][tuple(s_last)] - 1
                    if transition_matrix_copy[key][tuple(s_last)]<=0:
                        transition_matrix_copy[key].pop(tuple(s_last))   
            if  tuple(s) in list(transition_matrix_copy[tuple(s_last)].keys()):
                transition_matrix_copy[tuple(s_last)].pop(tuple(s))

        '''reduce the estimate occurance times of joint from each component in s and s_last''' 
        if tuple(s_last) in list(symbol_table_copy.keys()):
            symbol_table_copy[tuple(s_last)] = symbol_table_copy[tuple(s_last)]-symbol_table_copy[tuple(cat)]
            if symbol_table_copy[tuple(s_last)]<=0:
                symbol_table_copy.pop(tuple(s_last))
                marginals_copy.pop(tuple(s_last))
                transition_matrix_copy = pop_transition_matrix(transition_matrix_copy, tuple(s_last))

        if tuple(s) in list(symbol_table_copy.keys()):
            symbol_table_copy[tuple(s)] = symbol_table_copy[tuple(s)]-symbol_table_copy[tuple(cat)]
            if symbol_table_copy[tuple(s)] <= 0:
                symbol_table_copy.pop(tuple(s))
                marginals_copy.pop(tuple(s))
                transition_matrix_copy = pop_transition_matrix(transition_matrix_copy, tuple(s))

    '''Model has seen this chunk before'''
    if (tuple(cat) in list(symbol_table_copy.keys())) & (s_last != []) & (tuple(s) in list(symbol_table_copy.keys())):
        # reduce count from subtransition: 
        symbol_table_copy[tuple(cat)] = symbol_table_copy[tuple(cat)] + 1      
        if tuple(s_last) in list(symbol_table_copy.keys()):   
            symbol_table_copy[tuple(s_last)] = symbol_table_copy[tuple(s_last)] - 1
            if symbol_table_copy[tuple(s_last)]<=0:
                symbol_table_copy.pop(tuple(s_last))
                marginals_copy.pop(tuple(s_last))
                transition_matrix_copy = pop_transition_matrix(transition_matrix_copy, tuple(s_last))


        if tuple(s) in list(symbol_table_copy.keys()):   
            symbol_table_copy[tuple(s)] = symbol_table_copy[tuple(s)] - 1
            if symbol_table_copy[tuple(s)]<=0:
                symbol_table_copy.pop(tuple(s))
                marginals_copy.pop(tuple(s))
                transition_matrix_copy = pop_transition_matrix(transition_matrix_copy, tuple(s))


        if transition_matrix_copy != {}:
            # for key in list(transition_matrix.keys()):
                # look for transition to s_last and reduce count:
                # if tuple(s_last) in list(transition_matrix[key].keys()):
                #     transition_matrix[key][tuple(s_last)] = transition_matrix[key][tuple(s_last)] - 1 
                #     if transition_matrix[key][tuple(s_last)] <=0:
                #         transition_matrix[key].pop(tuple(s_last))
                # if tuple(cat) in list(transition_matrix[key].keys()):
                #     transition_matrix[key][tuple(cat)] = transition_matrix[key][tuple(cat)] + 1 
                # else: 
                #     transition_matrix[key][tuple(cat)] = 1
            if tuple(s_last) in list(transition_matrix_copy.keys()):
                if tuple(s) in list(transition_matrix_copy[tuple(s_last)].keys()):
                    transition_matrix_copy[tuple(s_last)][tuple(s)] = transition_matrix_copy[tuple(s_last)][tuple(s)] - 1
                    if transition_matrix_copy[tuple(s_last)][tuple(s)] <=0:
                        transition_matrix_copy[tuple(s_last)].pop(tuple(s))
    return symbol_table_copy,transition_matrix_copy, marginals_copy

def forgetting(symbol_table,deletion_threshold,marginals,transition_matrix,theta):
    """ discounting past observations"""
    for item in list(symbol_table.keys()): 
        symbol_table[item] = symbol_table[item]*theta # memory decays as a function of time 
        if symbol_table[item] < deletion_threshold: 
            symbol_table.pop(item)
            marginals.pop(item) # this assumes that marginals are up to date with items in symbol table
            transition_matrix = pop_transition_matrix(transition_matrix, tuple(item))
            # print("pop ", item, 'in marginals and symbol table because it is not used very often')
            if item == ():
                print("is an empty item in transition matrix key? ", item in list(transition_matrix.keys()))

    if transition_matrix != {}:
        for fromkey in list(transition_matrix.keys()):
            if transition_matrix[fromkey] != {}:
                for tokey in list(transition_matrix[fromkey].keys()):
                    transition_matrix[fromkey][tokey] = transition_matrix[fromkey][tokey]*theta
                    if transition_matrix[fromkey][tokey] < deletion_threshold:
                        transition_matrix[fromkey].pop(tokey)

    return transition_matrix,marginals,symbol_table


def rational_chunking(marginals, transition_matrix,s_last,s,w,Reward_structure = None,Punish_structure=None):
    '''
    Rational decision of chunking based on rewards and misses 
    '''
    cat =  list(s_last)+list(s)
    Expected_reward_cat = 0        
    if (s_last in list(marginals.keys()) and s_last in list(transition_matrix.keys())) and s in list(transition_matrix[s_last].keys()):
        sum_transition = 0
        for key in list(transition_matrix[s_last].keys()):
            sum_transition += transition_matrix[s_last][key]
        Expected_reward_cat = 0
        P_s_last = marginals[s_last]
        P_s_giv_s_last = transition_matrix[s_last][s]/sum_transition

        if Reward_structure!=None:
            R_chunk = 0
            # print('lists is', list(s))
            for i in list(s):
                R_chunk = R_chunk + Reward_structure[tuple([i])]

            Expected_reward_cat = P_s_giv_s_last*R_chunk

        else:
            Expected_reward_cat = P_s_giv_s_last*len(s)
        # print("expected reward positive part: ", Expected_reward_cat )

        if Punish_structure!=None:
            for key in list(transition_matrix[s_last].keys()):
                if key != s: # otherwise key will be overaccounted for. 
                    P_chunk = 0
                    missed_chunk = list(key)
                    for i in list(key):
                        P_chunk = P_chunk + Punish_structure[tuple([i])]
                    Expected_reward_cat = Expected_reward_cat + (-1)*w*P_chunk*transition_matrix[s_last][key]/sum_transition
        else: 
            Expected_reward_cat = Expected_reward_cat + (1 - P_s_giv_s_last)*(-1)*w*len(s)
    
    return Expected_reward_cat


def pop_transition_matrix(transition_matrix, element): 
    # pop an element out of a transition matrix
    if transition_matrix!={}:
        # element should be a tuple
        if element in list(transition_matrix.keys()):
            transition_matrix.pop(element)
            # print("pop ", item, 'in transition matrix because it is not used very often')
        for key in list(transition_matrix.keys()):
            if element in list(transition_matrix[key].keys()): # also delete entry in transition matrix
                transition_matrix[key].pop(element)
    return transition_matrix

def update_prediction_accuracy(predicted, prediction_accuracy,n):
    if predicted == True: 
        prediction_accuracy = (prediction_accuracy*n+1)/(n+1)
    else:
        prediction_accuracy = (prediction_accuracy*n)/(n+1)
    return prediction_accuracy

        
def update_recent_prediction_accuracy(predicted, prediction_accuracy,r):
    if predicted == True: 
        prediction_accuracy = (prediction_accuracy*r+1)/(r+1)
    else:
        prediction_accuracy = (prediction_accuracy*r)/(r+1)
    return prediction_accuracy    


def load_observation(o,marginals,this_sequence):
    # load proposed observation by looking into the future
    observation_candidates = []# find observation candidates that are consistant with remembered marginals
    observation_candidate_counts = []


    if list(marginals.keys()) == []:
        observation_candidates.append([this_sequence[o]])
        observation_candidate_counts.append(1.0)
    else: 
        for key in list(marginals.keys()):
            H_len = len(list(key))
            if this_sequence[o:o+H_len] == list(key):# observation is consistant with key 
                observation_candidates.append(list(key))
                observation_candidate_counts.append(marginals[key])
                # print('matching observation with key, ', key)
        if observation_candidates == []: # observation candidate not in observed in marginals before:
            observation_candidates.append([this_sequence[o]])
            observation_candidate_counts.append(1.0)
#     print('observation candidate is: ', observation_candidates, 'observation_candidate_count is', observation_candidate_counts)
#     final_candidate = sample_from_distribution(observation_candidates,observation_candidate_counts)


    # out of all observation candidates, select the longest one, 
    # or the most frequently used one in recent times,implicitly implemented with forgetting regime. 
    # print('observation candidates are:', observation_candidates)
    max_len = 0
    final_candidate = []
    final_candidate_count = []
    for candidate in observation_candidates:
        if len(candidate)>=max_len:
            max_len = len(candidate)
            final_candidate = candidate
    # print('final candidate is: ', final_candidate)
    #         print('indeed sampling from distribution')
    # states = observation_candidates
    # sample from an arbitrary distribution according to stimulus states and cumulative distribution function
    # prob =  [k/sum(observation_candidate_counts) for k in observation_candidate_counts]
    # return sample_from_distribution(states,prob)
    return final_candidate

def load_observation_past(o):
    """Load observation by looking into the past"""
    # current sequence element is sequence[o]
    observation_candidates = []# find observation candidates that are consistant with remembered marginals
    if list(marginals.keys()) == []:
        observation_candidates.append([sequence[o]])
    else: 
        for key in list(marginals.keys()):
            H_len = len(list(key))
            if sequence[o-H_len:o] == list(key):# observation is consistant with key 
                observation_candidates.append(list(key))
                # print('matching observation with key, ', key)
        
    # out of all observation candidates, select the longest one, 
    # or the most frequently used one in recent times,implicitly implemented with forgetting regime. 
    # print('observation candidates are:', observation_candidates)
    max_len = 0
    final_candidate = []
    for candidate in observation_candidates:
        if len(candidate)>=max_len:
            max_len = len(candidate)
            final_candidate = candidate
    # print('final candidate is: ', final_candidate)
    return final_candidate

def search_hypothesis(observation): 
    # search in hypothesis space on items that are consistant with observational data
    """
    STM(observation)[XXXXX]
    go backwards,find continuous subcomponent of a short term memory that matches up with the beginning
    There will be a lot of possible matches with chunks in short term memory, from different starting time and length. 
    # starting time, I mean starting in short term memory observation,
    and length, I mean the length i"""
    observation_candidates = []
    for m in symbol_table.keys():
        length_m = len(list(m))
        observation = sequence[o-length_m:o]
        # print('s_last ', observation, ', m ', m)
        if observation == list(m):
            observation_candidates.append(m)
    Entropy = []
    # print('s_last candidates are: ', observation_candidates)
    if observation_candidates == []:
        min_entropy_observation = sequence[o]
    else:
        for candidate in observation_candidates:
            Entropy.append(measure_entropy(transition_matrix, candidate))
        # print('choice entropy is: ', Entropy)
        min_entropy_choice = Entropy.index(min(Entropy))
        min_entropy_observation = list(symbol_table.keys())[min_entropy_choice]
    return min_entropy_observation



# helper functions
def eval_M_T(M,T,partitioned_sequence):
    """checked"""
    bag_of_chunks = M.index.tolist()
    
    for chunk in bag_of_chunks:
        parsed_chunk = json.loads(chunk)
        M.loc[chunk,'P'] = get_M_from_partitioned_sequence(parsed_chunk,partitioned_sequence)
    for chunk1 in bag_of_chunks:
        for chunk2 in bag_of_chunks:
            parsed_chunk1 = json.loads(chunk1)
            parsed_chunk2 = json.loads(chunk2)
            T.loc[chunk1,chunk2] = get_T_from_partitioned_sequence(parsed_chunk1,parsed_chunk2,partitioned_sequence)
    return M, T




def eval_M_T_from_original_sequence(M,T,original_sequence):
    bag_of_chunks = M.index.tolist()

    '''checked'''


    '''Get the estimated empirical probability of P(chunk2|chunk1),the probability of chunk2 followed by chunk1, 
    in the generated sequence
    In the case when chunk1 never occurs, output the probability of 0'''
    chunk1_count = 0
    chunk1_chunk2 = 0
    # the transition probability from chunk1 to chunk2
    # get P(chunk2|chunk1)
    not_over = True
    i = 0
    for candidate_chunk in partitioned_sequence: 
        if candidate_chunk == chunk1:
            chunk1_count +=1
            if i+1 < len(partitioned_sequence):
                candidate2 = partitioned_sequence[i+1]
                if candidate2 == chunk2:
                    chunk1_chunk2+=1
        i = i+1
    if chunk1_count>0:
        return chunk1_chunk2/chunk1_count
    else:
        return 0.0 # returns 0 if there is no occurrance for the first probabilility

    '''Get the estimated empirical probability of P(chunk),the probability of chunk2 followed by chunk1, 
    in the generated sequence
    In the case when chunk1 never occurs, output the probability of 0'''
    n_total = 0
    chunk1_count = 0
    for candidate_chunk in partitioned_sequence: 
        if candidate_chunk == chunk:
            chunk1_count +=1

    if chunk1_count>0:return chunk1_count/len(partitioned_sequence)
    else:return 0.0
    



# testing ring. 
def matrix_sample(M, T, s_l):
    
    """
    checked
    When it returns [], it means there is no prediction,
        otherwise, returns the predicted sequence of certain length as a list
        s_last: must be a list 
        returns [] when there is no information to sample. """
    if s_l == []: # no previous observation
        return [],0
    else:
        s_l = str(s_l) # M and T both have string indexing
        candidate_previous_chunks = T.index.tolist()
        if s_l not in candidate_previous_chunks:# no transition probability record, sample from marginal probability
            states = M.index.tolist()
            prob =  list(M['P'])
            state, prob = sample_from_list_distribution(states,prob)
            if state!=[]:    state = json.loads(state)
            return state, prob
        else:
            states = T.index.tolist()
            prob =  list(T.loc[s_l])
            state, prob = sample_from_list_distribution(states,prob)
            if state!=[]:    state = json.loads(state)
            return state, prob

def sample_from_list_distribution(states,prob):
    """
    checked
    sample from an arbitrary distribution 
    states: a list 
    prob: another list that contains the probability
    """
    if sum(prob) < 0.0001: # sum of probability equals 0
        return [],0
    else:
        prob =  [k/sum(prob) for k in prob]
        cdf = [0.0]
        for s in range(0, len(states)):
            cdf.append(cdf[s] + prob[s])
        k = np.random.rand()
        for i in range(1,len(states)+1):
            if (k>=cdf[i-1]):
                if (k < cdf[i]):
                    return states[i-1],prob[i-1]
    
    
    
def partition_seq(this_sequence,bag_of_chunks):
    '''checked'''
    # find the maximal chunk that fits the sequence
    # what to do when the bag of chunks does not partition the sequence?? 
    i = 0
    end_of_sequence = False
    partitioned_sequence = []
    
    while end_of_sequence == False:
        max_chunk = None
        max_length = 0 
        for chunk in bag_of_chunks: 
            this_chunk = json.loads(chunk)
            if this_sequence[i:i+len(this_chunk)] == this_chunk: 
                if len(this_chunk)> max_length:
                    max_chunk = this_chunk
                    max_length = len(max_chunk)

        if max_chunk == None:
            partitioned_sequence.append([this_sequence[i]])
            i = i + 1
        else:
            partitioned_sequence.append(list(max_chunk))
            i = i + len(max_chunk)

        if i >= len(this_sequence):end_of_sequence = True
            
    return partitioned_sequence
    
    
def get_T_from_partitioned_sequence(chunk1,chunk2,partitioned_sequence):
    '''checked'''


    '''Get the estimated empirical probability of P(chunk2|chunk1),the probability of chunk2 followed by chunk1, 
    in the generated sequence
    In the case when chunk1 never occurs, output the probability of 0'''
    chunk1_count = 0
    chunk1_chunk2 = 0
    # the transition probability from chunk1 to chunk2
    # get P(chunk2|chunk1)
    not_over = True
    i = 0
    for candidate_chunk in partitioned_sequence: 
        if candidate_chunk == chunk1:
            chunk1_count +=1
            if i+1 < len(partitioned_sequence):
                candidate2 = partitioned_sequence[i+1]
                if candidate2 == chunk2:
                    chunk1_chunk2+=1
        i = i+1
    if chunk1_count>0:
        return chunk1_chunk2/chunk1_count
    else:
        return 0.0 # returns 0 if there is no occurrance for the first probabilility


    
def get_M_from_partitioned_sequence(chunk,partitioned_sequence):
    '''checked'''


    '''Get the estimated empirical probability of P(chunk),the probability of chunk2 followed by chunk1, 
    in the generated sequence
    In the case when chunk1 never occurs, output the probability of 0'''
    n_total = 0
    chunk1_count = 0
    for candidate_chunk in partitioned_sequence: 
        if candidate_chunk == chunk:
            chunk1_count +=1

    if chunk1_count>0:return chunk1_count/len(partitioned_sequence)
    else:return 0.0
    
    

def matrix_load_observation(o,M,this_sequence):
    # find the maximally fitting chunk in the sequence
    # maximally fitting chunk or 
    if M.empty:
        return [this_sequence[o]]
    else: 
        chunks = M.index.tolist()
        max_chunk = None
        max_length = 0
        for chunk in chunks: 
            chunk = json.loads(chunk)
            C_len = len(chunk)
            if this_sequence[o:o+C_len] == chunk:# observation is consistant with key 
                if C_len >= max_length:
                    max_chunk = chunk
                    max_length = C_len
        if max_chunk == None:# no future prediction based on marginal
            return [this_sequence[o]]
        else:
            return max_chunk
    

# given a bag of chunks with their associated probability, 
# evaluate the loss function of this bag of chunks in the sequence.
# offline not from the past, but from the entire sequence. 

# a set of chunks partitions the entire sequence into chunks. 
# the chunk is identified by the maximal fitting element, 
# meaning that if there are two chunks that fits the description in the sequence, the biggest fitting chunk will be identified. 
# This creates a bias towards fitting bigger chunks.
# an alternative would be to fit the most likely chunk identified so far, 


    # evaluate loss function with respect to M and T. 
# L_acc, L_fast, M, T = evaluate_loss(bag_of_chunks, Acc = True, Fast = False)

def reorganize_M_T(M,T):
    chunks = M.index.tolist()
    for chunk in chunks:
        if M['P'][chunk] == 0:
            M = M.drop(chunk, axis=0)
            T = T.drop(chunk, axis=0)
            T = T.drop(chunk, axis=1)
    return M,T

def eval_L_Accurate(M,T):
    chunks = M.index.tolist()
    Expected_error = 0
    for chunk in chunks:
        p_ci = M['P'][chunk]
        conditional_mean_chunk = 0 # mean chunk size conditioned on the previous chunk observation
        for cj in chunks:
            p_cj_giv_ci = T.loc[chunk,cj]
            cj = json.loads(cj)
#             print(p_cj_giv_ci)
            conditional_mean_chunk += p_cj_giv_ci*len(cj)
        print('conditional mean chunk is ', conditional_mean_chunk)
        # once evaluated conditional mean chunks, go to evaluate the expected accuracy. 
        if conditional_mean_chunk > 0: # there is record of P(c|c_i)
            for cj in chunks:
                p_cj_giv_ci = T.loc[chunk,cj]
                cj = json.loads(cj)
                Expected_error += p_ci*p_cj_giv_ci*(1 - p_cj_giv_ci*len(cj)/conditional_mean_chunk)
                print('chunk ',chunk,'p_ci ',p_ci, 'cj ', cj,'p_cj_giv_ci ',p_cj_giv_ci, 'p_cj_giv_ci*len(cj)/conditional_mean_chunk',p_cj_giv_ci*len(cj)/conditional_mean_chunk, '(1 - p_cj_giv_ci*len(cj)/conditional_mean_chunk) ', (1 - p_cj_giv_ci*len(cj)/conditional_mean_chunk))

                # conditional_mean_chunk can be 0 when all entries of conditional probabilities are 0. 
    return Expected_error

def eval_L_Fast(M,T,RT_w, RT_b,c): # reevaluate the expected reaction time, every time 
    ceiling_rt_time = 0.8# second
    # reaction time between chunk: RT_b = -clog(P) + rt_b
    # reaction time within chunk: RT_w = rt_w r
    chunks = M.index.tolist()
    Expected_speed = 0
    for chunk in chunks:
        p_ci = M['P'][chunk]
        # once evaluated conditional mean chunks, go to evaluate the expected speed. 
        for cj in chunks:
            p_cj_giv_ci = T.loc[chunk,cj]
            if p_cj_giv_ci > 0.1:
                Expected_speed += p_ci*p_cj_giv_ci*(-np.log(p_cj_giv_ci)*c +RT_b + (len(cj)-1)*RT_w)/len(cj)
            else:
                Expected_speed += p_ci*p_cj_giv_ci*(ceiling_rt_time + RT_b + (len(cj)-1)*RT_w)/len(cj)
    return Expected_speed



def generate_ABCD_sequences(n_sample = 1100):
    H = 0.9
    M = 0.7
    stim_set = [(1,),(2,),(3,),(4,)]
    transition = {(1,): {(1,): (1-H)/3., (2,): H , (3,):(1-H)/3. ,(4,): (1-H)/3.}, 
                  (2,): {(1,): (1-M)/3. ,(2,): (1-M)/3., (3,): M ,(4,): (1-M)/3.},
                  (3,): {(1,): (1-H)/3. ,(2,): (1-H)/3. , (3,): (1-H)/3.,(4,): H},
                  (4,): {(1,): M ,(2,): (1-M)/3., (3,): (1-M)/3.,(4,): (1-M)/3.}}
    
    marginals = {(1,):0.25,(2,):0.25,(3,):0.25, (4,):0.25}
    sequence = []
    i = 1
    s_first,_ = list(sample_from_distribution(list(marginals.keys()), list(marginals.values())))
    sequence = sequence + s_first
    s_last = s_first
    while i < n_sample:
        i = i + 1
        new_sample,_ = sample(transition, marginals, s_last)
#         print('new sample is', new_sample)
        s_last = new_sample
        sequence = sequence + new_sample
    return sequence,stim_set


def round_sig(x, sig=2):
    if np.isclose(x,0):
        return 0.00
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)
    
    



def evaluate_loss_1(bag_of_chunks,RT_w,c):
    # goes over the sequence, and evaluate the loss function
    # bag_of_chunks: list of tuple
    if bag_of_chunks == []:
        L_acc = 1000
        L_fast = 1000
        M = pd.DataFrame([])
        T = pd.DataFrame([])
    else: 
        # initialize M and T
        M = pd.DataFrame(np.zeros([len(bag_of_chunks),1]))
        T = pd.DataFrame(np.zeros([len(bag_of_chunks),len(bag_of_chunks)]))
        M.index  = bag_of_chunks
        M.columns = ['P']
        T.columns = T.index = bag_of_chunks
        # partition the sequence into chunks 

        # find the maximal chunk that fits the sequence
        partitioned_sequence = partition_seq(this_sequence, bag_of_chunks)
        # evaluate M and T on the partitioned sequence. 
        M, T = eval_M_T(M,T,partitioned_sequence)
        L_rt = eval_rt(M,T,RT_w,c)# reevaluate the expected reaction time, every time 
    return L_rt, M, T




def eval_same_beginning(head, T, last_chunk):
    # given the observation of head, what is the probability of 1,2,3,4 following the head, according
    # to our internal model. 
    """head: list of chunk elements
        last_chunk: string""" 
    p_1 = 0
    p_2 = 0
    p_3 = 0
    p_4 = 0
    chunk1 = head+[1]
    chunk2 = head+[2]
    chunk3 = head+[3]
    chunk4 = head+[4]
#     print('last chunk ', last_chunk)
    """check the step-back of head in the from index of T"""
    """How many step-back, anyway? """

    for item in T.loc[last_chunk].index.tolist():
        Chunk = json.loads(item)
#         print(Chunk)
        if len(chunk1)<=len(Chunk):
            if Chunk[0:len(chunk1)] == chunk1:
                p_1+=T.loc[last_chunk][item]
            if Chunk[0:len(chunk1)] == chunk2:
                p_2+=T.loc[last_chunk][item]
            if Chunk[0:len(chunk1)] == chunk3:
                p_3+=T.loc[last_chunk][item]
            if Chunk[0:len(chunk1)] == chunk4:
                p_4+=T.loc[last_chunk][item]

    Sum = p_1+p_2+p_3+p_4
    if np.isclose(Sum,0): 
        return 0.25,0.25,0.25,0.25
    else:
        p_1 = p_1/Sum
        p_2 = p_2/Sum
        p_3 = p_3/Sum
        p_4 = p_4/Sum
        return p_1,p_2,p_3,p_4


def eval_same_beginning_M(head, M):
    # given the observation of head, what is the probability of 1,2,3,4 following the head, according
    # to our internal model. 
    p_1 = 0
    p_2 = 0
    p_3 = 0
    p_4 = 0
    chunk1 = head+[1]
    chunk2 = head+[2]
    chunk3 = head+[3]
    chunk4 = head+[4]
    for item in M.index.tolist():
        Chunk = json.loads(item)
        if Chunk[0:len(chunk1)] == chunk1:
            p_1+=M.loc[item]['P']
        if Chunk[0:len(chunk1)] == chunk2:
            p_2+=M.loc[item]['P']
        if Chunk[0:len(chunk1)] == chunk3:
            p_3+=M.loc[item]['P']
        if Chunk[0:len(chunk1)] == chunk4:
            p_4+=M.loc[item]['P']
    Sum = p_1+p_2+p_3+p_4
    p_1=p_1/Sum
    p_2 = p_2/Sum
    p_3 = p_3/Sum
    p_4 = p_4/Sum
    
    return p_1,p_2,p_3,p_4




# DFT
def DFT(p = [0.25,0.30,0.20,0.25],right_choice_index = 1,Plot = False):
    '''right choice: the index of the correct choice, matched with the probabilities'''
    n_step = 1000 # is in ms. 
    time = np.arange(0,n_step,1)
    path = np.zeros([4,n_step])
    winning_choice_index = None
    crossing_time = n_step
    v_instruction = 0.5 # parameter
    v_internal = (1.0-v_instruction)/3.0 # drift rate
    sigma = 0.6 # another parameter
    eps = 0.000000001
    b = 0 # boundary
    
    # initialization:
    for i in range(0, len(p)):
        if p[i] <=eps:path[i,0]= np.log(eps) 
        else: path[i,0]= np.log(p[i])
            
    for i in range(0, len(p)):  
        if i == right_choice_index:
            v = v_instruction# drift rate
        else:
            v = v_internal
        s = np.random.normal(v, sigma, n_step)
        for j in range(1,n_step):
            path[i,j]=path[i,j-1] +s[j]
            if path[i,j]>=0: 
                if j < crossing_time:
                    crossing_time = j
                    winning_choice_index = i
    # returns the first time it crosses 0
    if Plot: 
        plt.plot(time,path[0,:])
        plt.plot(time,path[1,:])
        plt.plot(time,path[2,:])
        plt.plot(time,path[3,:])
#         plt.ylim([0,-np.log(0.20)])
        plt.xlim([0,200])
        plt.legend(['1','2','3','4'])
    return crossing_time, winning_choice_index



def partition_seq_hastily(this_sequence,bag_of_chunks):
    c = 1
    eps = 0.01
    default_p = [eps,eps,eps,eps]
    '''checked'''
    # find the maximal chunk that fits the sequence
    # what to do when the bag of chunks does not partition the sequence?? 

    i = 0
    end_of_sequence = False
    partitioned_sequence = []
    true_chunk = None
    while end_of_sequence == False:
        for chunk in bag_of_chunks: 
            this_chunk = json.loads(chunk)
            if this_sequence[i] == this_chunk[0]: 
                partitioned_sequence.append(list(this_chunk))
                true_chunk = this_chunk
#             print('this_sequence', this_sequence[i:])
#             print('this chunk', this_chunk[0])
#         print(len(true_chunk), true_chunk)
        i = i + len(true_chunk)
        if i >= len(this_sequence):end_of_sequence = True


    M = pd.DataFrame(np.zeros([len(bag_of_chunks),1]))
    T = pd.DataFrame(np.zeros([len(bag_of_chunks),len(bag_of_chunks)]))
    M.index  = bag_of_chunks
    M.columns = ['P']
    T.columns = T.index = bag_of_chunks

    this_M, this_T = eval_M_T(M,T,partitioned_sequence)
    
#     this_M, this_T = eval_M_T_from_original_sequence(M,T,original_sequence)
#     print('this_sequence', this_sequence)
#     print('partitioned_sequence', partitioned_sequence)
#     print(this_M, this_T)

    correctness = []
    reacted_press = []
    rt = []
    choice_probability = []
    prev_obs = None
    i = 0

    # first item
    item = partitioned_sequence[0]

    p_1,p_2,p_3,p_4 = eval_same_beginning_M([], this_M)
    choice_p = max([p_1,p_2,p_3,p_4])
    instruction_index = this_sequence[i]-1
    choice_rt, choice_index = DFT(p = [p_1,p_2,p_3,p_4],right_choice_index = instruction_index)
    choice = choice_index + 1 
#     if choice_p<0.25:choice_p = 0.25
#     choice_rt = -c*np.log(choice_p) 
    if choice == this_sequence[i]:correctness.append(1)
    else:correctness.append(0) 
    
    rt.append(choice_rt)
    reacted_press.append(choice)
    choice_probability.append(choice_p)
    i = i + 1


    for j in range(1, len(item)):# choice probability always 1 inside a chunk
        choice = item[j]
        choice_p = 1
#         if choice_p<0.25:choice_p = 0.25
        choice_rt = -c*np.log(choice_p)
    
        choice_index = choice-1
        ps = default_p.copy()
        ps[choice_index] = 1 - 3*eps
        choice_p = 1
        instruction_index = this_sequence[i]-1
        choice_rt, choice_index = DFT(p = ps,right_choice_index = instruction_index)
        choice = choice_index + 1 
        if choice == this_sequence[i]:correctness.append(1)
        else:correctness.append(0)  
        rt.append(choice_rt)
        reacted_press.append(choice)
        choice_probability.append(choice_p)
        i = i + 1


    prev_obs = str(item)


    for item in partitioned_sequence[1:]:
        p_1,p_2,p_3,p_4 = eval_same_beginning([], this_T, prev_obs)
        choice_p = max([p_1,p_2,p_3,p_4])
        instruction_index = this_sequence[i]-1
        choice_rt, choice_index = DFT(p = [p_1,p_2,p_3,p_4],right_choice_index = instruction_index)
        choice = choice_index + 1 
        
        if i <= len(this_sequence)-1 and choice == this_sequence[i]:correctness.append(1)
        else:correctness.append(0) 
        rt.append(choice_rt)
        reacted_press.append(choice)
        choice_probability.append(choice_p)
        i = i + 1
        for j in range(1, len(item)):# choice probability always 1 inside a chunk
            choice = item[j]
            choice_index = choice-1
            ps = default_p.copy()
            ps[choice_index] = 1 - 3*eps
            choice_p = 1

#             if choice_p<0.25:choice_p = 0.25
#             choice_rt = -c*np.log(choice_p)
            if i <= len(this_sequence)-1 and choice == this_sequence[i]:
                instruction_index = this_sequence[i]-1
                choice_rt, choice_index = DFT(p = ps,right_choice_index = instruction_index)
                choice = choice_index + 1 
                correctness.append(1)
            else:correctness.append(0)  
            rt.append(choice_rt)
            reacted_press.append(choice)
            choice_probability.append(choice_p)
            i = i + 1
            
        prev_obs = str(item)
    end_len = len(this_sequence)
        # use the previous observation to predict the next elements.
    return correctness[0:end_len], rt[0:end_len], choice_probability[0:end_len],reacted_press[0:end_len],this_M, this_T
# print('this_sequence ',this_sequence)
# print('partitioned sequence', partitioned_sequence)
# print('correctnenss', correctness)
# print(rt)
# print('choice probability', choice_probability)
# print('reacted press',reacted_press)





    