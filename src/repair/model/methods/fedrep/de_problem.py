import time

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.problem import ElementwiseProblem

import numpy as np

import tensorflow as tf

from .SecureAggregation import aggregate_sum
from .utils import overwrite_weight, get_weight

class RepairProblem(ElementwiseProblem):

    def __init__(self, clients,model,weights,alpha=10,crypto=False):
        super().__init__(n_var=len(weights),
                         n_obj=1,
                         xl=-2.0,
                         xu=2.0)
        self.clients = clients
        self.initial_model = model
        self.weights = weights
        self.alpha = alpha
        self.crypto=crypto

    def prepare_location(self,values):
        """
        Prepares a list of weights to be copied to a model
        Parameters
        ----------
        values

        Returns
        -------

        """

        coordinates = self.weights

        res = [ [values[i]] + coordinates[i]  for i in range(len(values))]

        return res

    def _copy_location_to_weights(self, location, model):
        """Copy the candidate weights of the target locations to the model.

        :param location: consists of a neural weight value to mutate,
                          an index of a layer of the model,
                          and a neural weight position (i, j) on the layer
        :param model: subject DNN model
        :return: Modified DNN model
        """
        for w in location:
            # Parse location data
            val = w[0]
            layer_index = int(np.round(w[1]))
            nw_i = int(np.round(w[2]))
            nw_j = int(np.round(w[3]))

            # Set neural weight at given position with given value
            layer = model.get_layer(index=layer_index)
            weights = layer.get_weights()
            #weights = np.array(weights)
            i = w[2:]

            #weights[0][nw_i][nw_j] = val
            weights[0] = overwrite_weight(weights[0],i,val)
            layer.set_weights(weights)

        return model

    def score(self,model,imgs,labels):
        """
        Computes the score of one dataset
        Parameters
        ----------
        imgs
        labels

        Returns
        -------

        """

        total = 0

        predictions = model.predict(imgs,batch_size=16,verbose=0)
        cce = tf.keras.losses.CategoricalCrossentropy()

        for i,p in enumerate(predictions):

            predicted = np.argmax(p)
            true = np.argmax(labels[i])

            if predicted == true:
                total +=1
            else:
                #score = model.compute_loss(x=imgs[i],y=labels[i], y_pred=p)
                score = cce(labels[i], p).numpy()
                total += 1/(1+score)
        return total


    def _evaluate(self, x, out, *args, **kwargs):

        score_pool = []
        keyless_pool=[]

        # Loop through all clients
        for client in self.clients:

            input_neg = client.input_neg
            input_pos = client.input_pos

            begin=time.time()

            old_model = self.initial_model
            loc = self.prepare_location(x)
            new_model = self._copy_location_to_weights(loc,old_model)

            # Compute total score on neg and pos images
            score_neg = self.score(new_model,input_neg[0],input_neg[1])
            score_pos = self.score(new_model, input_pos[0], input_pos[1])

            # Combine into one
            combined = score_pos + self.alpha * score_neg
            crypto_sum = client.gen_encryption() if self.crypto else 0

            end = time.time()

            #print(f"Client {client.id}, time {end-begin}")

            score_pool.append(combined + crypto_sum)

        # Aggregate
        fitness = aggregate_sum(score_pool)

        # Multiply times -1 because minimizing - fitness == maximizing + fitness
        out["F"] = -1*fitness

    def get_initial(self,num_pop):
        """
        Get the original weight values, and sample the DE initialization form a
        Gaussian distribution
        Returns
        -------

        """

        target_layers = [x[0] for x in self.weights]
        target_layers = np.unique(target_layers)
        values = []

        layers = self.initial_model.layers

        means = []
        variances = []

        for l_id in target_layers:

            curr_layer = layers[l_id]
            curr_weights = curr_layer.get_weights()[0]

            # get mean and variance for initialization
            m = np.mean(curr_weights)
            v = np.var(curr_weights)

            print(m,v)

            coords = [x[1:] for x in self.weights if x[0]==l_id]

            means = means + [m]*len(coords)
            variances = variances + [v]*len(coords)

            for c in coords:
                #print(c,len(curr_weights),len(curr_weights[0]),len(curr_weights[0][0]))
                w_val = get_weight(curr_weights,c)
                values.append(w_val)

        print(f"Fetching initial values\n{values}")

        # Sample from Gaussian distribution
        mean = np.array(means)
        cov = np.identity(len(values))

        for i,v in enumerate(variances):
            cov[i][i]=v

        sampled = []
        generator = np.random.RandomState(42)

        for i in range(num_pop):
            s = generator.multivariate_normal(mean,cov)
            sampled.append(s)

        return sampled


class NonfedRepairProblem(RepairProblem):

    def __init__(self, input_neg,input_pos,model,weights,alpha=10,initial=None):
        super(RepairProblem,self).__init__(n_var=len(weights),
                         n_obj=1,
                         xl=-2.0,
                         xu=2.0)
        self.input_neg = input_neg
        self.input_pos = input_pos
        self.initial_model = model
        self.weights = weights
        self.alpha = alpha
        self.initial_pop = initial

    def _evaluate(self, x, out, *args, **kwargs):


        input_neg = self.input_neg
        input_pos = self.input_pos


        old_model = self.initial_model
        loc = self.prepare_location(x)
        new_model = self._copy_location_to_weights(loc, old_model)

        # Compute total score on neg and pos images
        score_neg = self.score(new_model, input_neg[0], input_neg[1])
        score_pos = self.score(new_model, input_pos[0], input_pos[1])

        # Combine into one
        fitness = score_pos + self.alpha * score_neg


        # Multiply times -1 because minimizing - fitness == maximizing + fitness
        out["F"] = -1 * fitness

    #def get_initial(self,num_pop):
    #    return self.initial_pop


