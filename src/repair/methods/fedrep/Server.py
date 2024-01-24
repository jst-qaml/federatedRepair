import numpy as np
from pymoo.termination.default import DefaultSingleObjectiveTermination

from .SecureAggregation import *
import keras
import tensorflow as tf

import itertools
import os
from pathlib import Path
import time

from repair.methods.arachnev2.arachne.run_localise import *
from .utils import extract_conv_layers, DE_alg_setup
from .de_problem import RepairProblem

from pymoo.optimize import minimize

class Server():

    def __init__(self) -> None:
        self.model = None
        self.clients = None

        # Lists to store locally computed gradient losses
        self.gl_res_pool_pos=[]
        self.gl_res_pool_neg=[]

        # Lists to store locally computed average activations
        self.act_res_pool_pos = []
        self.act_res_pool_neg = []

        # Lists to store locally computed average gradient of output w.r.t activation
        self.grad_res_pool_pos = []
        self.grad_res_pool_neg = []

        # Lists to store dataset size
        self.pos_sizes_list=[]
        self.neg_sizes_list=[]

        # Pool of fitness
        self.fitness_pool = None

        # Others
        self.arachne_v = 2
        self.model_dir = None
        self.gl_lab = None
        self.gl_pool = None
        self.act_lab = None
        self.act_pool = None
        self.go_lab = None
        self.go_pool = None
        self.conv_layers = [0,4,5,8,9]

    def connect_to_clients(self,clients):

        self.clients = clients


    def set_model(self,model,model_dir):

        #reshaped = keras.models.Model(
        #    model.layers[0].input, model.layers[layer_index].output
        #)
        #reshaped.compile(
        #    loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
        #)

        self.model = model
        self.model_dir = model_dir
        self.conv_layers = extract_conv_layers(model)

    def recv_local_FL_res(self,gl_neg,gl_pos,act_neg,act_pos,grad_neg,grad_pos):
        self.gl_res_pool_neg=gl_neg
        self.gl_res_pool_pos=gl_pos
        self.act_res_pool_neg=act_neg
        self.act_res_pool_pos=act_pos
        self.grad_res_pool_neg=grad_neg
        self.grad_res_pool_pos=grad_pos


    def recv_local_datasets_size(self,neg_sizes_list,pos_sizes_list):
        """
        Server receives the sizes of clients' local datasets
        Parameters
        ----------
        neg_sizes_list
        pos_sizes_list

        Returns
        -------

        """

        self.neg_sizes_list=neg_sizes_list
        self.pos_sizes_list=pos_sizes_list

    def recv_fitness(self,fitness_pool):
        self.fitness_pool = fitness_pool

    def ComputeAggregatedGradLoss_SingleWeight(self, lGradLossPos, lSizePos, lGradLossNeg, lSizeNeg):
        """
        Aggregates the gradient losses to compute the global gradient loss on a SINGLE WEIGHT
        """

        posGrad = AggregateAverage(lSizePos, lGradLossPos)
        negGrad = AggregateAverage(lSizeNeg, lGradLossNeg)


        return negGrad / (1 + posGrad)

    def ComputeAggregatedGradLos(self, localNegScores, localPosScores, neg_size, pos_size):
        """
        Aggregated the local Gradient Losses, for all weights.
        Parameters
        ----------

        Returns
        -------

        """

        # Sort each list of local results by weight location
        for local_neg_scores in localNegScores:
            local_neg_scores.sort(key = lambda tup:(tup[0], tup[1], tup[2]))

        for local_pos_scores in localPosScores:
            local_pos_scores.sort(key = lambda tup:(tup[0], tup[1], tup[2]))

        num_weights = len(localNegScores[0])
        grad_losses = []

        # Collect all individual scores for each weight, then aggreate
        for i in range(num_weights):

            neg_scores_single_w = [localNegScores[j][i][3] for j in range(len(localNegScores))]
            pos_scores_single_w = [localPosScores[j][i][3] for j in range(len(localPosScores))]
            w_index = localNegScores[0][i][0:3]
            grad_loss = self.ComputeAggregatedGradLoss_SingleWeight(pos_scores_single_w, pos_size, neg_scores_single_w, neg_size)

            grad_losses.append(w_index+[grad_loss])

        return grad_losses

    def ComputeAggregatedGradLossv2(self,gl_neg,gl_pos,sizes_neg,sizes_pos):

        layer_ids = gl_neg[0].keys()

        neg_pool = {}
        pos_pool = {}

        shapes = {}

        for l_id in layer_ids:
            neg_pool[l_id] = [gl_neg[i][l_id]['costs'] for i in range(len(gl_neg))]
            pos_pool[l_id] = [gl_pos[i][l_id]['costs'] for i in range(len(gl_pos))]
            shapes[l_id] = gl_pos[0][l_id]['shape']

        res_neg = AggregateAveragev2(sizes_neg,neg_pool)
        res_pos = AggregateAveragev2(sizes_pos,pos_pool)

        combined_gl = {}

        for l_id in layer_ids:
            combined_costs = res_neg[l_id] / (1 + res_pos[l_id])
            combined_costs = combined_costs.reshape(shapes[l_id])
            combined_gl[l_id] = {'shape': shapes[l_id],'costs':combined_costs}

        #combined_gl[4]['costs'] = np.sum(neg_pool[4],axis=0)

        return combined_gl


    def ComputeAggregatedAvgActivation(self,activationListPos,lSizePos, activationListNeg, lSizeNeg):
        """
        Aggregates the activations that were computed locally
        Parameters
        ----------
        activationListPos
        lSizePos
        activationListNeg
        lSizeNeg

        Returns
        -------

        """

        avg_act_neg = []
        avg_act_pos = []

        # First aggregate positive inputs' results
        for i in range(len(activationListPos[0])):

            activations = [activationListPos[j][i] for j in range(len(activationListPos))]

            avg = AggregateAverage(lSizePos,activations)

            avg_act_pos.append(avg)

        # Now aggregate negative inputs' results
        for i in range(len(activationListNeg[0])):
            activations = [activationListNeg[j][i] for j in range(len(activationListNeg))]

            avg = AggregateAverage(lSizeNeg, activations)

            avg_act_neg.append(avg)

        return avg_act_pos,avg_act_neg

    def aggregate_activationsv2(self, activationListPos, sizes_pos, activationListNeg, sizes_neg):

        layer_ids = activationListNeg[0].keys()

        neg_pool = {}
        pos_pool = {}

        for l_id in layer_ids:
            neg_pool[l_id] = [activationListNeg[i][l_id]['costs'] for i in range(len(activationListNeg))]
            pos_pool[l_id] = [activationListPos[i][l_id]['costs'] for i in range(len(activationListPos))]

            #print(activationListNeg[0][l_id]['costs'][0][0][0][0][0][0],activationListNeg[0][l_id]['costs'][0][0][0][0][0][0])

        res_neg = AggregateAveragev2(sizes_neg, neg_pool, use_abs=False)
        res_pos = AggregateAveragev2(sizes_pos, pos_pool)

        return res_neg,res_pos


    def NormalizeActivation_single(self,w,weights,activations,s):
        """
        Given a weight w, a list of weights in the same layer L+1 as w, and the activations of neurons from
        layer L, it computes the normalized activation needed for arachne's fw impact.
        Parameters
        ----------
        w: coordinates of the weight
        weights: list of
        activations

        Returns
        -------

        """

        i = w[1]
        j = w[2]

        w_ij = weights[i][j]
        o_i = activations[i]

        sum = 0

        if s is None:

            for k in range(len(activations)):
                sum += activations[k] * weights[k][j]
        else:
            sum = s

        return o_i * w_ij /sum, sum

    def NormalizeActivation(self,w_list,weights,activations):

        sum = None

        res = []

        for w in w_list:

            res_p, sum = self.NormalizeActivation_single(w,weights,activations,sum)

            res.append(w + tuple([res_p]))

        return res

    def NormalizeActivationv2(self,activations,denominators):

        res = {}

        for layer_idx in activations:

            num = activations[layer_idx]
            den = denominators[layer_idx]
            normalized = num/den

            res[layer_idx]=normalized

        return res

    #def reshape_activations(self,act,is_channel_first):
    #
    #    for l_idx in act:
    #        from_front = act[l_idx]
    #
    #        if is_channel_first:
    #            from_front = from_front.reshape(
    #                (n_output_channel, n_mv_0, n_mv_1, kernel_shape[0], kernel_shape[1], int(prev_output.shape[1])))
    #        else:  # channels_last
    #            from_front = from_front.reshape(
    #                (n_mv_0, n_mv_1, n_output_channel, kernel_shape[0], kernel_shape[1], int(prev_output.shape[-1])))


    def sum_FI_along_axis(self,fi_scores,l_idx,is_channel_first):
        """
        Sums the calculated FI scores to match arachne's definition.
        Parameters
        ----------
        fi_scores: The scores to sum
        l_idx: layer number
        is_channel_first: flag set to True if the model uses channel_first encoding

        Returns
        -------

        """

        # Sum only if the layer is convolutional
        if l_idx in self.conv_layers:

            if is_channel_first:
                fi_scores = np.sum(np.sum(fi_scores, axis = -1), axis = -1)
            else:
                fi_scores = np.sum(np.sum(fi_scores, axis=-2), axis=-2)

        return fi_scores


    def AggregatedGrad(self):

        grad_neg = []
        grad_pos = []

        num_clients = len(self.grad_res_pool_neg)
        num_grad = len(self.grad_res_pool_neg[0])

        for i in range(num_grad):

            single_neuron_grad_list = [self.grad_res_pool_neg[j][i] for j in range(num_clients)]
            score = AggregateAverage(self.neg_sizes_list,single_neuron_grad_list)
            grad_neg.append(score)

            single_neuron_grad_list = [self.grad_res_pool_pos[j][i] for j in range(num_clients)]
            score = AggregateAverage(self.pos_sizes_list, single_neuron_grad_list)
            grad_pos.append(score)

        return grad_neg, grad_pos

    def AggregatedGradv2(self,gout_neg,sizes_neg,gout_pos,sizes_pos):

        layer_ids = gout_neg[0].keys()

        neg_pool = {}
        pos_pool = {}

        for l_id in layer_ids:
            neg_pool[l_id] = [gout_neg[i][l_id]['costs'] for i in range(len(gout_neg))]
            pos_pool[l_id] = [gout_pos[i][l_id]['costs'] for i in range(len(gout_pos))]

        res_neg = AggregateAveragev2(sizes_neg, neg_pool)
        res_pos = AggregateAveragev2(sizes_pos, pos_pool)

        return res_neg, res_pos

    def fed_GradLoss(self):

        print("Server aggregating GL...", end="")

        if self.arachne_v == 1:
            global_GL = self.ComputeAggregatedGradLos(self.gl_res_pool_neg, self.gl_res_pool_pos, self.neg_sizes_list,
                                                         self.pos_sizes_list)
        elif self.arachne_v == 2:
            global_GL = self.ComputeAggregatedGradLossv2(self.gl_res_pool_neg, self.gl_res_pool_pos, self.neg_sizes_list,
                                                         self.pos_sizes_list)
        else:
            raise Exception(f"Arachne version {self.arachne_v} not supported or invalid")

        print("DONE")

        # Save aggregated scores
        self.save_FL(global_GL, None, 'Combined')

        return global_GL


    def fed_FwImpact(self, target_layer=None):
        """
        Computes federated forward impact
        """

        if self.arachne_v == 1:

            target_layer = self.model.layers[target_layer]

            weights = target_layer.get_weights()[0]
            w_shape = weights.shape

            w_list = list( itertools.product([0],list(range(w_shape[0])),list(range(w_shape[1])) ) )
            #print(w_list)

            print("Server aggregating activations...", end="")

            global_Activation_neg, global_Activation_pos = self.ComputeAggregatedAvgActivation(self.act_res_pool_pos, self.pos_sizes_list, self.act_res_pool_neg, self.neg_sizes_list)


            norm_neg = self.NormalizeActivation(w_list, weights, global_Activation_neg)

            norm_pos = self.NormalizeActivation(w_list, weights, global_Activation_pos)

            print("DONE")

            print("Server aggregating output gradients...", end="")

            grad_neg, grad_pos = self.AggregatedGrad()

            print("DONE")

            res = [ [ norm_neg[i][0], norm_neg[i][1], norm_neg[i][2], norm_neg[i][3]*grad_neg[norm_neg[i][2]]/(norm_pos[i][3]*grad_pos[norm_pos[i][2]]+1) ] for i in range(len(norm_neg))]

        elif self.arachne_v == 2:

            print("Server aggregating activations...", end="")


            act_neg, act_pos = self.aggregate_activationsv2(self.act_res_pool_pos, self.pos_sizes_list,
                                                            self.act_res_pool_neg, self.neg_sizes_list)

            print("DONE")

            print("Server aggregating output gradients...", end="")

            grad_neg, grad_pos = self.AggregatedGradv2(self.grad_res_pool_neg,self.neg_sizes_list,self.grad_res_pool_pos,self.pos_sizes_list)

            print("DONE")

            FI = {}

            for layer_idx in grad_neg:

                FI_neg = act_neg[layer_idx] * grad_neg[layer_idx]
                FI_pos = act_pos[layer_idx] * grad_pos[layer_idx]

                FI_neg = self.sum_FI_along_axis(FI_neg,layer_idx,is_channel_first=False)
                FI_pos = self.sum_FI_along_axis(FI_pos, layer_idx, is_channel_first=False)

                combined = FI_neg / (1 + FI_pos)
                shape = combined.shape

                print(layer_idx, combined.shape)

                FI[layer_idx]={'shape':shape,'costs':combined}

            self.save_FL(grad_loss=None,FI=FI,data_type="Combined")
            res = FI

        return res

    def gen_labels_scores(self,scores):

        labels = []
        score_pool = []

        for l_idx in scores:
            layer_scores = scores[l_idx]['costs']
            layer_scores = layer_scores.flatten()

            if score_pool is not None:
                score_pool = np.concatenate([score_pool,layer_scores])
            else:
                score_pool = np.copy(layer_scores)

            shape = scores[l_idx]['shape']

            for idx in itertools.product(*[range(s) for s in shape]):

                label = [l_idx] + list(idx)

                labels.append(label)

        return labels, score_pool

    def save_FL(self,grad_loss,FI,data_type='Negative',folder='Server'):
        """
        Saves the results of Server-side aggregation
        Parameters
        ----------
        grad_loss
        FI
        data_type

        Returns
        -------

        """

        path = Path(f"{folder}_{data_type}")

        if not os.path.exists(path):
            os.mkdir(path)

        if grad_loss is not None:

            # Save aggregated GL
            gl_path = path / "aggregated_GL"
            if not os.path.exists(gl_path):
                os.mkdir(gl_path)

            for l_idx in grad_loss:
                layer_path = gl_path / f"layer_{l_idx}"
                scores = grad_loss[l_idx]['costs']
                np.save(layer_path,scores)

        if FI is not None:

            # Save aggregated FI
            fi_path = path / "aggregated_FI"
            if not os.path.exists(fi_path):
                os.mkdir(fi_path)

            for l_idx in FI:
                layer_path = fi_path / f"layer_{l_idx}"
                scores = FI[l_idx]['costs']
                np.save(layer_path, scores)

    def two_dim_Pareto(self,x_scores,y_scores,labels=None):
        """
        Given the x-scores, y-scores, and the coordinates of corresponding weight, the method
        computes the Pareto front.
        Parameters
        ----------
        x_scores
        y_scores
        coords

        Returns
        -------

        """

        if labels is None:
            labels = list(range(len(x_scores)))

        pool = [(labels[i],x_scores[i],y_scores[i]) for i in range(len(x_scores))]

        print("Done preparing pool.")

        # sort by x score
        if self.arachne_v == 1:
            pool = sorted( pool, key=lambda x: x[1][3])
        else:
            pool = sorted(pool, key=lambda x: x[1],reverse=True)

        y_max = -1
        res = []
        scores = []

        for i in range(len(pool)):

            y_score = pool[i][2][3] if self.arachne_v == 1 else pool[i][2]

            if y_score >= y_max:
                res.append(pool[i][0])
                scores.append( (pool[i][1], pool[i][2]) )
                y_max = y_score

        return res, scores


    def localize(self):
        """
        Aggregate individual FL results received from clients.
        Returns
        -------

        """

        t_begin = time.time()

        # Compute gradient loss and forward impact
        gl = self.fed_GradLoss()
        fw_imp = self.fed_FwImpact(-1)

        t_end = time.time()
        delta = (t_end - t_begin) * 1000

        print(f"Server took {delta} ms to aggregate FL results.")

        assert len(gl) == len(fw_imp), f"Cannot compute pareto front! GL contains {len(gl)} scores but FI has {len(fw_imp)}!"

        print("Extracting pareto front...",end="")

        if self.arachne_v == 1:
            labels = [ x[0:3] for x in gl]
            pareto = self.two_dim_Pareto(gl,fw_imp,labels)
        elif self.arachne_v == 2:

            gl_lab, gl_pool = self.gen_labels_scores(gl)
            fi_lab, fi_pool = self.gen_labels_scores(fw_imp)

            pareto, scores = self.two_dim_Pareto(gl_pool,fi_pool,gl_lab)

        print("DONE")

        return pareto, (gl_lab,gl_pool), (fi_lab,fi_pool)


    def optimize(self,weights,num_gen=2,num_pop=10):
        """
        Aggregated PSO to optimize weights.
        Returns
        -------

        """

        print(f"Server optimizing: {num_gen} generations, {num_pop} individuals")

        # Get a problem instance
        rep_problem = RepairProblem(
            clients=self.clients,
            model=self.model,
            weights=weights,
            crypto=False
        )

        # Get initial population
        initial_pop=rep_problem.get_initial(num_pop)

        # Setup the algorithm
        algorithm = DE_alg_setup(initial_pop,population_size=num_pop)

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
            problem=rep_problem,
            algorithm=algorithm,
            termination=termination,
            seed=42,
            verbose=True,
            save_history=True
        )

        print("OPTIMAL SOLUTION")
        print(f"Found weights: {res.X}")
        print(f"Fitness: {res.F}")

        f=open("optimization_stack.txt","a")
        f.write(f"{res.X}\n")
        f.close()

        #print(res.history)

        return (res.X,res.F,initial_pop,res.history)

