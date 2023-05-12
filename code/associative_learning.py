def associative_learning(seq, theta=1):
    '''First order associative learning model'''
    # theta: forgetting rate
    transition={}
    for i in [1,2,3,4]:
        transition[i] = {}
        for j in [1,2,3,4]:
            transition[i][j] = 0
    p = [0.25] # initialize the first probability to be flat
    
    for i in range(1,len(seq)):
        prev = seq[i-1]
        curr = seq[i]
        transition[prev][curr] = (transition[prev][curr] + 1)*theta
        p.append(transition[prev][curr]/sum(transition[prev].values()))
        
    return p
        