import numpy as np
import sys
import os

model='VGG16'
classes = [0,1,2,3,6,7,8,9,12]
n_clients = [1,2,5,10]

base_folder = sys.argv[1]

def aggregate(folder):
    print("Aggregating in folder",folder)

    neg_dir= f"{folder}/Negative_results"
    pos_dir = f"{folder}/Positive_results"

    n_clients = len(os.listdir(neg_dir))

    activations_neg=[]
    activations_pos = []
    outs_neg=[]
    outs_pos = []
    gls_neg=[]
    gls_pos = []

    for cl in range(n_clients):
        client_dir_neg = f"{folder}/Negative_results/Client_{cl}"
        client_dir_pos = f"{folder}/Positive_results/Client_{cl}"

        # Combine GL
        gl_neg = np.load(f"{client_dir_neg}/GL/layer_22.npy")
        gl_pos = np.load(f"{client_dir_pos}/GL/layer_22.npy")

        gls_pos.append(gl_pos)
        gls_neg.append(gl_neg)

        # Combine FI
        act_neg = np.load(f"{client_dir_neg}/Act/layer_22.npy")
        act_pos = np.load(f"{client_dir_pos}/Act/layer_22.npy")
        out_neg = np.load(f"{client_dir_neg}/Out/layer_22.npy")
        out_pos = np.load(f"{client_dir_pos}/Out/layer_22.npy")

        activations_neg.append(act_neg)
        activations_pos.append(act_pos)
        outs_neg.append(out_neg)
        outs_pos.append(out_pos)

    aggregatedActNeg = np.sum(activations_neg,axis=0)
    aggregatedActPos = np.sum(activations_pos,axis=0)
    aggregatedOutNeg = np.sum(outs_neg,axis=0)
    aggregatedOutPos = np.sum(outs_pos,axis=0)
    FI_neg = aggregatedActNeg*aggregatedOutNeg
    FI_pos = aggregatedActPos*aggregatedOutPos
    FI = FI_neg/(1+FI_pos)

    print(FI.shape)

    aggregatedGlNeg = np.sum(gls_neg,axis=0)
    aggregatedGlPos = np.sum(gls_pos,axis=0)
    GL = aggregatedGlNeg/(1+aggregatedGlPos)

    print(GL.shape)

    np.reshape(FI,GL.shape)

    return aggregatedGlNeg, aggregatedGlPos, FI_neg, FI_pos

for c in classes:

    for n in n_clients:
        folder=f"{base_folder}/Cl{c}_Mod{model}_N{n}"

        GLneg, GLpos, FIneg,FIpos = aggregate(folder)

        np.save(f"{base_folder}/glNeg_Cl{c}_Mod{model}_N{n}",GLneg)
        np.save(f"{base_folder}/fiNeg_Cl{c}_Mod{model}_N{n}", FIneg)
        np.save(f"{base_folder}/glPos_Cl{c}_Mod{model}_N{n}", GLpos)
        np.save(f"{base_folder}/fiPos_Cl{c}_Mod{model}_N{n}", FIpos)