import itertools
import random
from pathlib import Path
import csv

import pickle
import numpy as np
import tensorflow as tf
import os

from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.optimize import minimize

from repair.core import method
from repair.core.dataset import RepairDataset


from .Client import Client
from .Server import Server

from repair.methods.arachnev2.arachne.run_localise import *
from .utils import *

from .SecureAggregation import DH_key_exchange

from .de_problem import NonfedRepairProblem

distributions={
    0:[0.8, 0.05,0.05,0.05,0.05],
    1:[0.6, 0.1,0.1,0.1,0.1],
    2:[0.4,0.15, 0.15,0.15,0.15],
    4:[0.3,0.175, 0.175,0.175,0.175]
}


class FedRep(method.RepairMethod):

    def __init__(self,n_clients=2):

        self.server = Server()
        self.client_pool = []
        self.num_clients = n_clients
        self.neg_sizes = {}
        self.pos_sizes = {}
        self.num_grad = None
        self.num_particles = 100
        self.num_iterations = 100
        self.num_input_pos_sampled = 200
        self.velocity_phi = 4.1
        self.min_iteration_range = 10
        self.target_layer = 13
        self.output_files = set()
        self.batch_size = 32
        self.arachne_v = 2
        self.distr_id=None

    def set_options(self, **kwargs):
        """Set options."""
        if "num_grad" in kwargs:
            self.num_grad = kwargs["num_grad"]
        if "num_particles" in kwargs:
            self.num_particles = kwargs["num_particles"]
        if "num_iterations" in kwargs:
            self.num_iterations = kwargs["num_iterations"]
        if "num_input_pos_sampled" in kwargs:
            self.num_input_pos_sampled = kwargs["num_input_pos_sampled"]
        if "velocity_phi" in kwargs:
            self.velocity_phi = kwargs["velocity_phi"]
        if "min_iteration_range" in kwargs:
            self.min_iteration_range = kwargs["min_iteration_range"]
        if "target_layer" in kwargs:
            self.target_layer = int(kwargs["target_layer"])
        if "batch_size" in kwargs:
            self.batch_size = int(kwargs["batch_size"])
        if "model_dir" in kwargs:
            self.model_dir = kwargs["model_dir"]
        if "n_clients" in kwargs:
           n = int(kwargs['n_clients'])
           self.num_clients = n if n>0 else 1
        if "distr_id" in kwargs:
            n = int(kwargs['distr_id'])
            self.distr_id = n

    def save_weights(self, weights, output_dir: Path):
        """Save neural weight candidates.

        Parameters
        ----------
        weights
            Weights to be saved
        output_dir : Path
            Path to directory to save weights

        """
        with open(output_dir / "weights.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(weights)
            self.output_files.add(output_dir / "weights.csv")

    def load_weights(self, weights_dir: Path):
        """Load neural weight candidates.

        Parameters
        ----------
        weights_dir : Path
            Path to directory containing 'weights.csv'

        Returns
        -------
        Neural weight candidates

        """
        candidates = []

        try:
            with open(weights_dir / "weights.csv") as f:
                reader = csv.reader(f)
                for row in reader:
                    w_list = [int(x) for x in row]
                    candidates.append(w_list)
        except:
            print(f"{weights_dir} not found. Fetching weights later")
        return candidates

    def load_input_neg(self, neg_dir: Path):
        """Load negative inputs.

        Parameters
        ----------
        neg_dir : Path
            Path to directory containing negative dataset

        Returns
        -------
        input_neg
            Loaded negative dataset

        """
        return RepairDataset.load_repair_data(neg_dir)

    def load_input_pos(self, pos_dir: Path):
        """Load positive inputs.

        Parameters
        ----------
        pos_dir : Path
            Path to directory containing positive dataset

        Returns
        -------
        input_pos
            Loaded positive dataset

        """
        return RepairDataset.load_repair_data(pos_dir)

    def create_clients(self,model,input_neg,input_pos,model_dir):
        """
        Instantiates clients
        Parameters
        ----------
        model
        input_neg
        input_pos

        Returns
        -------

        """

        for i in range(self.num_clients):
            c = Client(i,input_neg[i],input_pos[i],model,model_dir=model_dir)
            self.client_pool.append(c)
            self.neg_sizes[i] = len(c.input_neg[0])
            self.pos_sizes[i] = len(c.input_pos[0])


    def split_dataset(self,dataset,k,distr=None):
        """
        Splits a dataset into k datasets.

        Parameters
        ----------
        dataset

        Returns
        -------

        """
        
        imgs = dataset[0]
        labels = dataset['labels']

        dataset_size = len(imgs)

        if distr is None:
            individual_size =dataset_size // k

        sizes = []
        tot = dataset_size
        index=0

        while tot > 0 and index<k:

            if not distr is None:
                print(index,distr)
                individual_size = int(distr[index]*dataset_size)
                index+=1

            if tot >= individual_size:
                sizes.append(individual_size)
                tot -= individual_size
            else:
                sizes[-1] += tot
                tot -= tot

        if sum(sizes) <dataset_size:
            sizes[-1] += -sum(sizes) + dataset_size

        print("Distributing data to each client",sizes)

        split_imgs = []
        split_labels = []


        imgs_local = []
        labels_local = []
        im_index = 0
        lab_index = 0
        j = 0

        for img in imgs:
            imgs_local.append(img)
            im_index +=1


            if im_index >= sizes[j]:
                split_imgs.append(imgs_local)
                imgs_local = []
                im_index = 0
                j+=1

        j = 0
        for lab in labels:

            labels_local.append(lab)
            lab_index += 1

            if lab_index >= sizes[j]:
                split_labels.append(labels_local)
                labels_local = []
                lab_index = 0
                j+=1

        datasets = [[np.asarray(split_imgs[i]),np.asarray(split_labels[i])] for i in range(k)]

        #d1 = datasets[0][0]
        #d2 = dataset[0]

        #for i in range(len(d1)):
        #
        #    assert (d1[i]==d2[i]).all(), f"Position {i}"

        return datasets

    def key_agreements(self,client_pool):

        size = len(client_pool)


        for i in range(size):
            for j in range(size-i):

                if i != j:

                    signs = ['+', '-']

                    key = random.uniform(0,10)

                    sign = signs[random.randint(0,1)]
                    signs.remove(sign)
                    c1 = client_pool[i]
                    c2 = client_pool[j]
                    c1.add_key(key,sign,c2.id)
                    c2.add_key(key,signs[0],c1.id)

    def save_nonfed_FL_pos(self,output_dir,nonfed_scores):

        path = output_dir

        if not os.path.exists(path):
            os.mkdir(path)

        path = Path(path)/Path("Positive_results")

        if not os.path.exists(path):
            os.mkdir(path)

        grad_loss_path = path / "nonfed_scores"
        if not os.path.exists(grad_loss_path):
            os.mkdir(grad_loss_path)

        for l_idx in nonfed_scores:
            layer_path = grad_loss_path / f"layer_{l_idx}"
            scores = nonfed_scores[l_idx]['costs']
            scores = np.array(scores)
            shape = nonfed_scores[l_idx]['shape']
            np.save(layer_path,scores)

            f = open(grad_loss_path /f"shape_{l_idx}.dat" ,"wb")
            pickle.dump(shape,f)
            f.close()


    def load_nonfed_FL_pos(self, output_dir):

        res = {}

        path = Path(output_dir)/ "Positive_results" / "nonfed_scores"

        for file_path in os.listdir(path):

            #print(os.path.isfile(file_path) , ("layer_" in file_path))
            if ("layer_" in file_path):
                layer_num = file_path.split("_")[1].split(".")[0]
                layer_num = int(layer_num)

                data = np.load(path/Path(file_path),allow_pickle=True)

                shape_path = path/f"shape_{layer_num}.dat"

                f = open(shape_path,"rb")
                shape = pickle.load(f)
                f.close()

                res[layer_num] = {}
                res[layer_num]['shape'] = shape
                res[layer_num]['costs'] = data

        return res

    def non_fed_localize(self,input_neg, input_pos,model_dir,output_dir="",target_layer_list=[]):
        """
        Run plain FL to compare later
        Returns
        -------

        """

        # Run arachnev2 on the whole dataset for reference

        # First on negative inputs
        X = input_neg[0]
        y = input_neg[1]

        t_w = get_target_weights(None, model_dir)
        #t_w.pop(1)
        #t_w = { 13: t_w[13] }
        t_w = {l:t_w[l] for l in target_layer_list}

        neg_indices = list(range(len(X)))
        nf_res_neg = compute_FI_and_GL(X, y,
                                   indices_to_target=neg_indices,
                                   target_weights=t_w,
                                   is_multi_label=True,
                                   path_to_keras_model=model_dir,
                                   federated=False)

        # Repeat for pos inputs
        X = input_pos[0]
        y = input_pos[1]

        #t_w = get_target_weights(None, model_dir)
        #t_w.pop(1)
        #t_w = { 13: t_w[13] }

        pos_indices = list(range(len(X)))

        # Perform the computation only if it was not done before
        pos_fl_path = Path(f"{output_dir}/Positive_results/nonfed_scores")
        if not os.path.exists(pos_fl_path):
            print(f"File {pos_fl_path} not found. Going for first computation")
            nf_res_pos = compute_FI_and_GL(X, y,
                                   indices_to_target=pos_indices,
                                   target_weights=t_w,
                                   is_multi_label=True,
                                   path_to_keras_model=model_dir,
                                   federated=False)
            self.save_nonfed_FL_pos(output_dir,nf_res_pos)
        else:
            nf_res_pos = self.load_nonfed_FL_pos(output_dir)

        for tl_idx in nf_res_neg:
            print("LAYER", tl_idx)
            print(f"\tGRAD LOSS: shape {nf_res_neg[tl_idx]['shape']}")


        # Separate GL and FI scores
        nf_res_gl = {}
        nf_res_fi = {}

        for l_idx in nf_res_neg:
            shape = nf_res_neg[l_idx]['shape']
            costs = nf_res_neg[l_idx]['costs']
            costs_pos = nf_res_pos[l_idx]['costs']

            gl_scores = []
            fi_scores = []

            for idx in range(len(costs)):

                gl_fi_neg = costs[idx]
                gl_fi_pos = costs_pos[idx]
                gl = gl_fi_neg[0]/(1+gl_fi_pos[0])
                fi = gl_fi_neg[1]/(1+gl_fi_pos[1])

                gl_scores.append(gl)
                fi_scores.append(fi)

            nf_res_gl[l_idx] = {'shape':shape,'costs': np.asarray(gl_scores)}
            nf_res_fi[l_idx] = {'shape':shape, 'costs': np.asarray(fi_scores)}

        #nf_res_gl = nf_res[0]
        #nf_res_out = nf_res[3]

        # Process and save GL
        nf_labels, nf_scores = self.server.gen_labels_scores(nf_res_gl)

        nf_gl_scores = [nf_scores[i] for i in range(len(nf_scores) )]

        nf_gl_data = [((nf_gl_scores[i], 0), nf_labels[i]) for i in range(len(nf_labels))]

        f = open('non_fed_GL.txt','w')
        f.write(str(nf_gl_data))
        f.close()

        # Process and save FI
        old_labels = nf_labels
        nf_labels, nf_scores = self.server.gen_labels_scores(nf_res_fi)

        assert old_labels == nf_labels

        nf_fi_scores = [nf_scores[i] for i in range(len(nf_scores))]

        nf_fi_data = [((nf_fi_scores[i], 0), nf_labels[i]) for i in range(len(nf_labels))]

        f = open('non_fed_FI.txt', 'w')
        f.write(str(nf_fi_data))
        f.close()

        return nf_gl_data, nf_fi_data

    def non_fed_optimize(self,model,weights,input_neg,input_pos,num_gen,num_pop,initial):
        """
        Runs the optimization in a non-federated fashion
        Parameters
        ----------
        model
        weights
        input_neg
        input_pos
        num_gen
        num_pos

        Returns
        -------

        """

        problem = NonfedRepairProblem(input_neg,input_pos,model,weights,initial=initial)

        # Get initial population
        initial_pop = problem.get_initial(num_pop)

        # Setup the algorithm
        algorithm = DE_alg_setup(initial_pop, population_size=num_pop)

        # Fix the termination condition
        termination = DefaultSingleObjectiveTermination(
            xtol=1e-8,
            cvtol=1e-6,
            ftol=1e-6,
            period=10,
            n_max_gen=num_gen,
            n_max_evals=100000
        )

        res = minimize(
            problem=problem,
            algorithm=algorithm,
            termination=termination,
            seed=42,
            verbose=True
        )

        print("OPTIMAL NON-FED SOLUTION")
        print(f"Found weights: {res.X}")
        print(f"Fitness: {res.F}")

        return (res.X, res.F)


    def localize(self, model, input_neg, output_dir: Path, verbose=1,input_pos=None,model_dir=None):
        """
        The clients perform local FL, the server aggregates them
        Parameters
        ----------
        model Model to perform FL
        input_neg Dataset of negative inputs
        output_dir Directory to write outputs
        verbose

        Returns
        -------

        """

        # Split negative and positive inputs among all clients and create them
        input_neg_split = self.split_dataset(input_neg,self.num_clients)
        input_pos_split = self.split_dataset(input_pos,self.num_clients)
        self.create_clients(model,input_neg_split,input_pos_split,model_dir)
        self.key_agreements(self.client_pool)

        if self.distr_id is not None:
            distribution = distributions[self.distr_id]
        else:
            distribution = None

        # Lists to store locally computed gradient losses
        gl_res_pool_pos=[]
        gl_res_pool_neg=[]

        # Lists to store locally computed average activations
        act_res_pool_pos = []
        act_res_pool_neg = []

        # Lists to store locally computed average output gradients
        grad_res_pool_pos = []
        grad_res_pool_neg = []

        # We localize in the last layer
        last = len(model.layers) - 1

        # Hydra model does not have weights in the very last layer
        if "hydra" in model_dir:
            last -=1
        tgt_layers = [last]

        # Each client computes the gradient loss and activations locally
        for c in self.client_pool:

            print("Client #", c.id, "is localizing")

            if self.arachne_v == 1:
                scores = c.local_GradientLoss()
                gl_res_pool_neg.append(scores[0])
                gl_res_pool_pos.append(scores[1])
                neg_activations, pos_activations = c.local_Activation()
                act_res_pool_neg.append(neg_activations)
                act_res_pool_pos.append(pos_activations)
                neg_grad, pos_grad = c.local_OutputGrad()
                grad_res_pool_neg.append(neg_grad)
                grad_res_pool_pos.append(pos_grad)

            elif self.arachne_v == 2:

                grad_loss_neg, activ_neg, grad_out_neg, grad_loss_pos, activ_pos, grad_out_pos = c.local_FI_and_GL(output_dir,tgt_layers)
                gl_res_pool_neg.append(grad_loss_neg)
                gl_res_pool_pos.append(grad_loss_pos)
                act_res_pool_neg.append(activ_neg)
                act_res_pool_pos.append(activ_pos)
                grad_res_pool_neg.append(grad_out_neg)
                grad_res_pool_pos.append(grad_out_pos)

            else:
                raise Exception(f"Arachne version {self.arachne_v} not supported or invalid!")

        # Server receives the original model
        self.server.set_model(model,model_dir)

        neg_sizes_list = np.array( [ self.neg_sizes[id] for id in self.neg_sizes] )
        pos_sizes_list = np.array( [self.pos_sizes[id] for id in self.pos_sizes] )

        # Server receives datasets sizes
        self.server.recv_local_datasets_size(neg_sizes_list,pos_sizes_list)

        # Server receives FL results from each client
        self.server.recv_local_FL_res(gl_res_pool_neg,gl_res_pool_pos,act_res_pool_neg,act_res_pool_pos,grad_res_pool_neg,grad_res_pool_pos)

        if self.num_clients > 1:
            # Server aggregates the results
            suspicious_weights, scores_gl_fed, scores_fi_fed = self.server.localize()

            self.save_weights(suspicious_weights,output_dir)
        else:
            # Run plain FL, save results
            non_fed_GL,non_fed_FI = self.non_fed_localize(input_neg, input_pos, model_dir,output_dir,tgt_layers)

        ## Test that GL and FI were aggregated correctly
        #fed_GL = [( (scores_gl_fed[1][i],0),scores_gl_fed[0][i]) for i in range(len(non_fed_GL))]
        #fed_FI = [( (scores_fi_fed[1][i],0),scores_fi_fed[0][i]) for i in range(len(non_fed_FI))]
        #
        #f = open('fed_GL.txt', 'w')
        #f.write(str(fed_GL))
        #f.close()
        #
        #f = open('fed_FI.txt', 'w')
        #f.write(str(fed_FI))
        #f.close()
        #
        ## Compare raw scores
        #gl_comp = compare_scores(non_fed_GL, fed_GL,threshold=0.01)
        #fi_comp = compare_scores(non_fed_FI, fed_FI,threshold=0.01)

            # Compute non-federated pareto front
            keys = [non_fed_GL[i][1] for i in range(len(non_fed_GL))]
            nf_x_gl = [s[0][0] for s in non_fed_GL]
            nf_y_fi = [s[0][0] for s in non_fed_FI]
            non_fed_pf, non_fed_pf_scores = self.server.two_dim_Pareto(nf_x_gl,nf_y_fi,keys)

            print("Pareto front(Non-fed)\n",non_fed_pf,"\n",non_fed_pf_scores)

            print(f"Saving to {output_dir}")
            self.save_weights(non_fed_pf,output_dir)

        ## Compare with federated pareto front
        #I = [x for x in non_fed_pf if x in suspicious_weights]
        #U = non_fed_pf
        #
        #for x in suspicious_weights:
        #    if x not in U:
        #        U.append(x)
        #jacobi_index = len(I)/len(U)
        #
        #print(f"Jacobi index of pareto fronts: {jacobi_index}")
        #
        #f = open(f"{output_dir}/fl_comparison.txt","w")
        #f.write(f"GL results: failed {gl_comp} % of comparisons\n")
        #f.write(f"FI results: failed {fi_comp} % of comparisons\n")
        #f.write(f"JacobiIndex: {jacobi_index}\n")
        #f.write(f"Federated PF:\n{suspicious_weights}\n")
        #f.write(f"Non-Federated PF:\n{non_fed_pf}\n")
        #f.close()

        return 0

    def optimize(

        self,
        model,
        model_dir: Path,
        weights,
        input_neg,
        input_pos,
        output_dir: Path,
        verbose=1,
        num_gen=2,
        num_pop=8
    ):


        weights = self.load_weights(output_dir)
        print(weights)
        #weights = [[0,1,2],[13,3,4]]

        if self.distr_id is not None:
            distribution = distributions[self.distr_id]
        else:
            distribution = None

        # Split negative and positive inputs among all clients and create them
        input_neg_split = self.split_dataset(input_neg, self.num_clients,distribution)
        input_pos_split = self.split_dataset(input_pos, self.num_clients,distribution)
        self.create_clients(model, input_neg_split, input_pos_split, model_dir)

        # Server establishes connection with clients
        self.server.connect_to_clients(self.client_pool)

        # Clients run key agreement
        DH_key_exchange(self.client_pool,output_dir=output_dir)

        # Server receives the original model
        self.server.set_model(model, model_dir)

        if self.num_clients > 1:

            # Optimize in a federated way
            fed_values, fed_fitness, initial_pop, fed_history = self.server.optimize(weights,num_gen=num_gen,num_pop=num_pop)

            f = open(f"{output_dir}/optimization_history.log","wb")
            pickle.dump(fed_history[-1],f)
            f.close()
        else:
            # Optimize in a non-fed way
            non_fed_values, non_fed_fitness =self.non_fed_optimize(
                model,
                weights,
                input_neg,
                input_pos,
                num_gen,
                num_pop,
                initial=None
            )

        ## Compare
        #fed_values = np.array(fed_values)
        #non_fed_values = np.array(non_fed_values)
        #l2norm = np.linalg.norm(fed_values-non_fed_values)
        #gap = fed_fitness - non_fed_fitness
        #
        #print(f"L2 norm of found solutions: {l2norm}")
        #print(f"Gap between fed and non-fed fitness: {gap}")

    def evaluate(
            self,
            dataset,
            model_dir: Path,
            target_data,
            target_data_dir: Path,
            positive_inputs,
            positive_inputs_dir: Path,
            output_dir: Path,
            num_runs,
    ):
        print("Method for evaluation still under development")