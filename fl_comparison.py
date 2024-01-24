import numpy as np
from pathlib import Path

model='VGG19'

datasets_sizes={
    'VGG16': {
        0:67,
        1:352,
        2:333,
        3:74,
        6:148,
        7:104,
        8:108,
        9:121,
        12:463,
        -1:11516
    },

    'VGG19': {
        0:227,
        1:457,
        2:456,
        3:107,
        6:411,
        7:156,
        8:266,
        9:381,
        12:841,
        -1:9984
    }
}

classes = [0,1,2,3,6,7,8,9,12]

base_dir = "./"

def compute_scores(cl,model,n_client):
    non_fed_fi_pos = np.load(Path(base_dir) / Path(f"results_folder/fiPos_Cl{cl}_Mod{model}_N{n_client}.npy"))
    non_fed_gl_pos = np.load(Path(base_dir) / Path(f"results_folder/glPos_Cl{cl}_Mod{model}_N{n_client}.npy"))
    non_fed_fi_neg = np.load(Path(base_dir) / Path(f"results_folder/fiNeg_Cl{cl}_Mod{model}_N{n_client}.npy"))
    non_fed_gl_neg = np.load(Path(base_dir) / Path(f"results_folder/glNeg_Cl{cl}_Mod{model}_N{n_client}.npy"))

    non_fed_fi_pos = non_fed_fi_pos / (datasets_sizes[model][-1]) ** 2
    non_fed_fi_neg = non_fed_fi_neg / (datasets_sizes[model][cl]) ** 2
    non_fed_gl_pos = non_fed_gl_pos / (datasets_sizes[model][-1])
    non_fed_gl_neg = non_fed_gl_neg / (datasets_sizes[model][cl])

    nonfed_gl = non_fed_gl_neg / (1 + non_fed_gl_pos)
    nonfed_fi = non_fed_fi_neg / (1 + non_fed_fi_pos)

    return nonfed_gl, nonfed_fi
def comparison():

    for cl in classes:

        print("Class",cl)

        #Get non-fed results
        nf_gl, nf_fi = compute_scores(cl,model,1)

        for cl_num in [2,5,10]:
            f_gl, f_fi = compute_scores(cl,model,cl_num)

            diff_gl = np.abs(f_gl/nf_gl - 1)
            diff_fi = np.abs(f_fi / nf_fi - 1)

            diff_gl = diff_gl > 0.01
            diff_fi = diff_fi > 0.01

            gl_count = np.count_nonzero(diff_gl)
            fi_count = np.count_nonzero(diff_fi)

            print(cl_num,gl_count/len(diff_gl),fi_count/len(f_gl))

comparison()
