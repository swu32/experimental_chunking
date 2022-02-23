# filter out participants with mean reaction time  
# input: data frame 
def data_filtering(df):

    exclude_participants = set()
    for ID in np.unique(df['id']):
        for n_block in range(1, 11):
            blockwise_mean_RT = np.mean(df.loc[(df['block']==n_block) & (df['id'] == ID),['timecollect']])
            blockwise_mean_acc = np.mean(df.loc[(df['block']==n_block) & (df['id'] == ID),['correctcollect']])
            if blockwise_mean_RT['timecollect']>1000 or blockwise_mean_acc['correctcollect'] < 0.9:
                exclude_participants.add(ID)
    df.loc[df['id'].isin(exclude_participants)]
    dff = df.loc[~df['id'].isin(exclude_participants)]
    return dff





from matplotlib import pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
#------------------------------------------------------------
# Learn the best-fit GaussianMixture models
#  Here we'll use scikit-learn's GaussianMixture model. The fit() method
#  uses an Expectation-Maximization approach to find the best
#  mixture of Gaussians for the data

# fit models with 1-10 components
def learn_gaussian_mixture(X, plot = True,n_mixture = 3):
    N = [n_mixture] #np.arange(1, 11)
    models = [None for i in range(len(N))]

    for i in range(len(N)):
        models[i] = GaussianMixture(N[i]).fit(X)

    # compute the AIC and the BIC
    AIC = [m.aic(X) for m in models]
    BIC = [m.bic(X) for m in models]

    #------------------------------------------------------------
    # Plot the results
    #  We'll use three panels:
    #   1) data + best-fit mixture
    #   2) AIC and BIC vs number of components
    #   3) probability that a point came from each component
    if plot: 
        fig = plt.figure(figsize=(15, 5))
        fig.subplots_adjust(left=0.12, right=0.97,
                            bottom=0.21, top=0.9, wspace=0.5)


        # plot 1: data + best-fit mixture
        ax = fig.add_subplot(131)
        M_best = models[np.argmin(AIC)]

        x = np.linspace(6, 8, 1000)
        # print(x)
        logprob = M_best.score_samples(x.reshape(-1, 1))
        responsibilities = M_best.predict_proba(x.reshape(-1, 1))
        print('shape of responsibilities: ', responsibilities.shape)
        pdf = np.exp(logprob)
        pdf_individual = responsibilities * pdf[:, np.newaxis]

        ax.hist(X, 30, density=True, histtype='stepfilled', alpha=0.4)
        ax.plot(x, pdf, '-k')
        ax.plot(x, pdf_individual, '--k')
        ax.text(0.04, 0.96, "Best-fit Mixture",
                ha='left', va='top', transform=ax.transAxes)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$p(x)$')


        # plot 2: AIC and BIC
        ax = fig.add_subplot(132)
        ax.plot(N, AIC, '-k', label='AIC')
        ax.plot(N, BIC, '--k', label='BIC')
        ax.set_xlabel('n. components')
        ax.set_ylabel('information criterion')
        ax.legend(loc=2)


        # plot 3: posterior probabilities for each component
        ax = fig.add_subplot(133)
        print('responsibilities', responsibilities)

        p = responsibilities
        p = p[:, (1, 0)]  # rearrange order so the plot looks better
        p = p.cumsum(1).T

        ax.fill_between(x, 0, p[0], color='gray', alpha=0.3)
        ax.fill_between(x, p[0], p[1], color='gray', alpha=0.5)
        ax.fill_between(x, p[1], 1, color='gray', alpha=0.7)
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1)
        ax.set_xlabel('$x$')
        ax.set_ylabel(r'$p({\rm class}|x)$')

        ax.text(-5, 0.3, 'class 1', rotation='vertical')
        ax.text(0, 0.5, 'class 2', rotation='vertical')
        ax.text(3, 0.3, 'class 3', rotation='vertical')

        # plt.show()
        plt.savefig('gaussian_mixture_example.png')
    return AIC, BIC, models



# code that does the gaussian mixture model classicition
def mixture_gaussian_classification(time_collect, n_mixture = 2):
    # group level behavioral comparison: 
    within_chunk = np.zeros([1*1000], dtype =bool)
    p_within_chunk = np.zeros([1*1000])
    
    AIC, BIC, model = lgm(time_collect,n_mixture = n_mixture)
    model = model[0]
    within_chunk_model_index = np.argmin(model.means_)
    reaction_time = np.array(time_collect)
    responsibilities = model.predict_proba(reaction_time.reshape(-1, 1))# likelihood of belonging to which gaussian mixture
    prediction = model.predict(reaction_time.reshape(-1, 1))# likelihood of belonging to which gaussian mixture
    within_chunk = (prediction == within_chunk_model_index)
    p_within_chunk = responsibilities[:,within_chunk_model_index]
    return p_within_chunk, within_chunk


def lgm(X,n_mixture = 3): # a shortened version of the learn gaussian mixture model. 
    N = [n_mixture] #np.arange(1, 11)
    models = [None for i in range(len(N))]

    for i in range(len(N)):
        models[i] = GaussianMixture(N[i]).fit(X)

    # compute the AIC and the BIC
    AIC = [m.aic(X) for m in models]
    BIC = [m.bic(X) for m in models]

    return AIC, BIC, models


from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def fit_with_gaussian(timecollect, color='green'):
    '''Fit a gaussian to data'''
    datos = timecollect
    (mu, sigma) = norm.fit(datos)

    # the histogram of the data
    n, bins, patches = plt.hist(datos, bins = 60, density=True, facecolor= color, alpha=0.75)

    # add a 'best fit' line
    y = norm.pdf( bins, mu, sigma)
    l = plt.plot(bins, y, 'r--', linewidth=2)
    # scipy.stats.norm.pdf i

    #plot
    plt.xlabel('ms')
    plt.ylabel('Probability')
    plt.title(r'$\mathrm{Histogram\ of\ Reaction\ Time:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
    plt.grid(True)

    plt.show()


def GMC(data):
    """usage: processing dataframe in csv files and add a column of withinchunk and p_within_chunk in the corresponding dataframe"""
    df = data
    import random
    # group level behavioral comparison: 
    within_chunk = np.zeros([len(np.unique(df['id']))*1000], dtype =bool)
    p_within_chunk = np.zeros([len(np.unique(df['id']))*1000])
    i = 0
    for ID in np.unique(df['id']):    
        # select participants with a particular ID 
        this_subject = df[df['id'] == ID] # set to one particular subject

        X = np.array(this_subject['timecollect'][this_subject['timecollect']<1000])# when using np.log, make sure nan and infinity are excluded. 
        X = X.reshape(-1,1)

        AIC, BIC, model = learn_gaussian_mixture(X, plot = False, n_mixture = 3)
        model = model[0]

        reaction_time = np.array(this_subject['timecollect'])
        responsibilities = model.predict_proba(reaction_time.reshape(-1, 1))# likelihood of belonging to which gaussian mixture
        prediction = model.predict(reaction_time.reshape(-1, 1))# likelihood of belonging to which gaussian mixture

        within_chunk_model_index = np.argmin(model.means_)
        within_chunk[i*1000:(i+1)*1000] = (prediction == within_chunk_model_index)
        p_within_chunk[i*1000:(i+1)*1000] = responsibilities[:,within_chunk_model_index]

        i = i+1
        # prediction: true for within chunk, and false for between chunk
        # responsibilities: probability of being a wihtin chunk item 

    # the probability of being a within chunk element
    # print(len(responsibilities[:,0]))# trial length x mixture array
    df['withinchunk'] = pd.Series(np.ravel(list(within_chunk)))
    df['p_within_chunk'] = pd.Series(np.ravel(list(p_within_chunk)))
    return df

def identify_press(dff, keydff, idx):
    """input: the row of a dataframe """
    # identify chunks
    if dff.loc[idx]['userpress'] == keydff.loc[idx][2]:
        return 1
    if dff.loc[idx]['userpress'] == keydff.loc[idx][5]:
        return 2
    if dff.loc[idx]['userpress'] == keydff.loc[idx][8]:
        return 3
    if dff.loc[idx]['userpress'] == keydff.loc[idx][11]:
        return 4
    else:
        return None
    
def identify_key_seq(subject_id, blocks, dataframe):
    '''returns the translated keyinstruction, key press, in [1,2,3,4] integers across the defined range of blocks '''
    instruction = []
    press = []
    thisrt = []
    for block in blocks:
        for trial in range(1,101):
            index = dataframe.index[(dataframe['id'] == subject_id) & (dataframe['trialcollect'] == trial) & (dataframe['block'] == block)].tolist()[0]
            keypress = identify_press(dataframe, keydff, index)
            inst = identify_inst(dataframe, keydff, index)
            press.append(keypress)
            instruction.append(inst)
            thisrt.append(dataframe[(dataframe['id'] == subject_id) & (dataframe['trialcollect'] == trial) & (dataframe['block'] == block)]['timecollect'].tolist()[0])
    return instruction, press, thisrt


    
def identify_inst(dff, keydff, idx):
    """input: the row of a dataframe """
    # identify chunks
    if dff.loc[idx]['instructioncollect'] == keydff.loc[idx][2]:
        return 1
    if dff.loc[idx]['instructioncollect'] == keydff.loc[idx][5]:
        return 2
    if dff.loc[idx]['instructioncollect'] == keydff.loc[idx][8]:
        return 3
    if dff.loc[idx]['instructioncollect'] == keydff.loc[idx][11]:
        return 4
    else:
        return None
    

def process_experiment1_into_chunks_by_condition(pathex1):
    dff = pd.read_csv(pathex1)
    subject_ids = list(np.unique(dff['id']))
    keydff = dff['keyassignment'].str.split(r"\[|]|'|,", expand = True) # spilt key index to identify keys

    indep_chunks = {}
    c3_chunks = {}# dictionary to record chunk 3
    c2_chunks = {}# dictionary to record chunk2

    idx = 0
    for subject_id in subject_ids: 
        if dff[dff['id'] == subject_id]['condition'].tolist()[0] == 2: # [12] [3] [4] reward'
            chunk_collect = c2_chunks    
        if dff[dff['id'] == subject_id]['condition'].tolist()[0] == 1: # [123] reward'
            chunk_collect = c3_chunks    
        if dff[dff['id'] == subject_id]['condition'].tolist()[0] == 0: # [1] [2] [3] [4]
            chunk_collect = indep_chunks
        # process chunks and load them in the chunk collect
        name = str(subject_id)
        chunk_collect[name] = []
        # iterate in every block, single chunk are recorded by iteself.
        for block in range(1,11):
            i = 0
            for trial in range(1,101):
                chunk = []# initiate with an empty chunk 
                this_row = dff[(dff['id'] == subject_id) & (dff['trialcollect'] == trial) & (dff['block'] == block)]
                if this_row['withinchunk'].tolist()[0] == False:
                    # search back in time to find another false:
                    index = dff.index[(dff['id'] == subject_id) & (dff['trialcollect'] == trial) & (dff['block'] == block)].tolist()[0]
                    keypress = identify_press(dff, keydff, index)
                    chunk.append(keypress)
                    next_trial = trial + 1
                    if next_trial <=100: 
                        nextrow = dff[(dff['id'] == subject_id) & (dff['trialcollect'] == next_trial) & (dff['block'] == block)]
                        while nextrow['withinchunk'].tolist()[0] == True and next_trial<=100:
                            index = dff.index[(dff['id'] == subject_id) & (dff['trialcollect'] == next_trial) & (dff['block'] == block)].tolist()[0]
                            keypress = identify_press(dff, keydff, index)
                            chunk.append(keypress)
                            next_trial = next_trial + 1
                            nextrow = dff[(dff['id'] == subject_id) & (dff['trialcollect'] == next_trial) & (dff['block'] == block)]
                            if len(nextrow)==0:
                                break
                    chunk_collect[name].append(chunk)
                    i = i + len(chunk)
                else:
                    pass
                
    return indep_chunks, c2_chunks, c3_chunks




def process_experiment2_into_chunks_by_condition(pathex2):
    dff = pd.read_csv(pathex2)

    subject_ids = list(np.unique(dff['id']))
    keydff = dff['keyassignment'].str.split(r"\[|]|'|,", expand = True) # spilt key index to identify keys

    f_chunks = {}# dictionary to record fast group chunks
    a_chunks = {}# dictionary to record accurate group chunks

    idx = 0
    for subject_id in subject_ids: # iterate through subject to extract their learned chunks
        if dff[dff['id'] == subject_id]['condition'].tolist()[0] == 1: # accuracy
            chunk_collect = a_chunks    
        if dff[dff['id'] == subject_id]['condition'].tolist()[0] == 0: # fast
            chunk_collect = f_chunks
        # process chunks and load them in the chunk collect
        name = str(subject_id)
        chunk_collect[name] = []
        for block in range(1,11):# iterate in every block, single chunk are recorded by iteself.
            i = 0
            for trial in range(1,101):
                chunk = []# initiate with an empty chunk 
                this_row = dff[(dff['id'] == subject_id) & (dff['trialcollect'] == trial) & (dff['block'] == block)]
                if this_row['withinchunk'].tolist()[0] == False:
                    # search back in time to find another false:
                    index = dff.index[(dff['id'] == subject_id) & (dff['trialcollect'] == trial) & (dff['block'] == block)].tolist()[0]
                    #print('index is,', index, 'trial is ', trial)
                    keypress = identify_press(dff, keydff, index)
                    chunk.append(keypress)
                    next_trial = trial + 1
                    if next_trial <=100: 
                        nextrow = dff[(dff['id'] == subject_id) & (dff['trialcollect'] == next_trial) & (dff['block'] == block)]
                        while nextrow['withinchunk'].tolist()[0] == True and next_trial<=100:
                            index = dff.index[(dff['id'] == subject_id) & (dff['trialcollect'] == next_trial) & (dff['block'] == block)].tolist()[0]
                            keypress = identify_press(dff, keydff, index)
                            chunk.append(keypress)
                            next_trial = next_trial + 1
                            nextrow = dff[(dff['id'] == subject_id) & (dff['trialcollect'] == next_trial) & (dff['block'] == block)]
                            if len(nextrow)==0:
                                break
                    chunk_collect[name].append(chunk)
                    i = i + len(chunk)
                else:
                    pass   
                
    return f_chunks, a_chunks



