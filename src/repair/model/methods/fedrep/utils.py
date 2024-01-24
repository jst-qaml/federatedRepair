import numpy as np

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS

from randomgen import ChaCha
from numpy.random import Generator

from tqdm import tqdm

def compare_scores(non_fed, fed, threshold=0.15,index=0):
    """
    Verifies if scores has been correctly aggregated
    Parameters
    ----------
    non_fed List of non_federated scores = [ ((GL,FI),(weight_key))]
    fed List of federated scores = [ ((GL,FI),(weight_key))]
    Returns
    -------

    """

    fed.sort(key= lambda x: x[1])
    non_fed.sort(key = lambda x:x[1])
    sorted_f = fed
    sorted_nf = non_fed

    failed = []
    ratios= []

    labels_f = [x[1] for x in sorted_f]
    labels_nf = [x[1] for x in sorted_nf]

    #assert labels_nf == labels_f, f"Different labels detected!"

    assert len(sorted_f) == len(sorted_nf), f"Mismatching lengths! FED :{len(sorted_f)} vs NON-FED: { len(sorted_nf)}"

    for i in tqdm(range(len(sorted_nf))):

        nf = sorted_nf[i][0][0]
        f = sorted_f[i][0][0]

        #assert sorted_nf[i][1] == sorted_f[i][1], f"Unmatching keys in pos {i}: FED: {sorted_f[i][1]} vs NON-FED: {sorted_f[i][1]}"
        #if sorted_nf[i][1] != sorted_f[i][1]:
        #    print(f"Unmatching keys in pos {i}: FED: {sorted_f[i][1]} vs NON-FED: {sorted_f[i][1]}")
        ratio=nf/f if (nf,f) != (0,0) else 1

        close = np.abs(ratio - 1) < threshold
        if not close:
            failed.append((i,ratio,f,nf))
            ratios.append(ratio)

    print(f"FAILED {len(ratios)/len(sorted_f) * 100} % of comparisons")

    print(failed[0:10])

    return failed, len(ratios)/len(sorted_f) * 100

def extract_conv_layers(model):
    """
    Given a keras model, it returns the list of its convolutional layers (indexes)
    Parameters
    ----------
    model   The model of interest

    Returns
    -------

    """

    conv = []

    for i,l in enumerate(model.layers):
        type = l.__class__.__name__

        if type == 'Conv2D':
            conv.append(i)

    if 1 in conv:
        conv.remove(1)
    
    return conv

def overwrite_weight(weights,index,val):
    """
    Ovewrites weight with a given index in a list of weights
    Parameters
    ----------
    weights
    index

    Returns
    -------

    """



    if len(index)==2:
        weights[index[0]][index[1]]=val
    elif len(index)==3:
        weights[index[0]][index[1]][index[2]]=val
    elif len(index)==4:
        weights[index[0]][index[1]][index[2]][index[3]]=val
    elif len(index)==5:
        weights[index[0]][index[1]][index[2]][index[3]][index[4]]=val
    elif len(index)==6:
        weights[index[0]][index[1]][index[2]][index[3]][index[4]][index[5]]=val
    else:
        raise Exception(f"Number of indexes not supported: {len(index)}")

    return weights


def get_weight(weights, index):
    """
    Ovewrites weight with a given index in a list of weights
    Parameters
    ----------
    weights
    index

    Returns
    -------

    """

    if len(index) == 2:
        return weights[index[0]][index[1]]
    elif len(index) == 3:
        return weights[index[0]][index[1]][index[2]]
    elif len(index) == 4:
        return weights[index[0]][index[1]][index[2]][index[3]]
    elif len(index) == 5:
        return weights[index[0]][index[1]][index[2]][index[3]][index[4]]
    elif len(index) == 6:
        return weights[index[0]][index[1]][index[2]][index[3]][index[4]][index[5]]
    else:
        raise Exception(f"Number of indexes not supported: {len(index)}")

def DE_alg_setup(initial,population_size=10):
    """
    Prepares the Diff. Evolution algorithm
    Returns
    -------

    """

    print(f"Initializing to population {initial}")

    np.random.seed(1)

    # We use the same params
    # CR=0.7
    # F sampled randomly form (0.5,1) (dither)
    algorithm = DE(
        pop_size=population_size,
        sampling=np.array(initial),
        variant="DE/rand/1/bin",
        CR=0.7,
        dither="vector",
        jitter=False,
    )

    return algorithm

def chacha20_setup(keys):
    """
    Creates an array of chacha20 PRNGs seeded with given keys
    Parameters
    ----------
    keys

    Returns
    -------

    """

    prngs = {}

    for k in keys:

        seed = abs(keys[k])
        gen = ChaCha(seed=seed)
        gen = Generator(gen)
        prngs[k] = gen

    return prngs

