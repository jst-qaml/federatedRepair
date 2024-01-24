import numpy as np
from Crypto.Util.number import getPrime,getRandomInteger

from sys import getsizeof
from pathlib import Path
from time import time

def AggregateAverage(lSize, lVal):
    """
    Given a list of sizes and a list of values computed locally, it computes the federated average
    """

    assert len(lSize) == len(lVal), f"Input lists have sizes {len(lSize)} and {len(lVal)}"

    s = 0
    n = 0

    for i in range(len(lVal)):
        s += lVal[i]
        n += lSize[i]

    return s / n

def AggregateAveragev2(lSize,lVal,use_abs=True):
    """
    Method for aggregation of results produced by Arachnev2
    Parameters
    ----------
    lSize
    lVal

    Returns
    -------

    """

    res = {}

    total_size = np.sum(lSize)

    for layer in lVal:
        layer_values = lVal[layer]
        summed = np.sum(layer_values,axis=0)

        if use_abs:
            res[layer] = np.abs(summed / total_size)
        else:
            res[layer] = summed / total_size

    return res

def aggregate_sum(lVal):
    """
    Aggregates scores by summing them
    Parameters
    ----------
    lVal

    Returns
    -------

    """
    return np.sum(lVal)

def DH_key_exchange(clients,output_dir=""):
    """
    Given a pool of clients, simulates a Diffie-Hellmann key exchange
    between any pair of clients
    Parameters
    ----------
    clients

    Returns
    -------

    """

    size = len(clients)

    for c in clients:
        c.reset_keys()

    n_bits = 256
    g = 2
    q = getPrime(n_bits)
    p = 2*q+1

    print(f"Running Diffie-Hellman key-exchange with params \ng={g},\np={p} ({n_bits} bits)", end="...")

    tot_size = 0

    for i in range(size):
        for j in range(i+1,size):

            c1 = clients[i]
            c2 = clients[j]

            x = getRandomInteger(n_bits)
            y = getRandomInteger(n_bits)

            m1 = pow(g,x,p)
            m2 = pow(g,y,p)

            key1 = pow(m1,y,p)
            key2 = pow(m2,x,p)

            assert key1==key2, f"Error in key exchange between clients {i} and {j}"

            tot_size += getsizeof(m1)*2

            c1.add_key(key1,'+',j)
            c2.add_key(key1, '-', i)

    for c in clients:
        print(c.key_pool)
        c.gen_prngs()

    # Log the message exchanged
    path = Path(output_dir) / Path("message_log.txt")
    f = open(path, "a")
    f.write(f"[{time()}] Clients exchange {tot_size} bytes for key exchange\n")
    f.close()


    print("Key exchange DONE!")
