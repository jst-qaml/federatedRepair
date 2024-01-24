import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
from tqdm import trange

from keras import backend as K

import os
from sys import getsizeof
from pathlib import Path
from struct import unpack
import time

from repair.methods.arachne import arachne
from repair.methods.arachnev2.arachne.run_localise import *

from .utils import chacha20_setup

class Client(arachne.Arachne):

    def __init__(self,id,input_neg,input_pos,model,model_dir=None) -> None:
        self.id = id
        self.input_neg = input_neg
        self.input_pos = input_pos
        self.neg_size = len(self.input_neg[0])
        self.pos_size = len(self.input_pos[0])
        print(f"CLIENT SIZES {self.neg_size},{self.pos_size}")
        self.model = model
        self.num_grad=None
        self.num_particles = 100
        self.num_iterations = 100
        self.num_input_pos_sampled = 200
        self.velocity_phi = 4.1
        self.min_iteration_range = 10
        self.target_layer = None
        self.output_files = set()
        self.batch_size = 32
        self.key_pool={}
        self.encrypt = False
        self.model_dir=model_dir
        self.alpha=0.1
        self.prngs = []

    def reset_keys(self):
        self.key_pool={}

    def add_key(self,key,sign,id):

        if sign=='-':
            key = -1*key
        self.key_pool[id]=key

    def gen_prngs(self):
        """
        Generates and initialize local PRNGs
        Returns
        -------

        """

        self.prngs = chacha20_setup(self.key_pool)

    def gen_encryption(self):
        """
        Generates the encryption mask
        Returns
        -------

        """

        sum = 0

        for k in self.key_pool:

            key = self.key_pool[k]
            prng = self.prngs[k]

            num = prng.bytes(8)
            num = unpack("d",num)[0]

            if key > 0:
                sum += num
            else:
                sum -= num

        return sum



    def rescale_scores(self,score_pool,neg_flag):
        """
        Given a set of scores, it multiplies them by the dataset size to prepare them for aggregation.
        Parameters
        ----------
        scores

        Returns
        -------

        """

        if neg_flag:
            size = self.neg_size
        else:
            size = self.pos_size

        # Multiply by dataset size
        for l_idx in score_pool:
            scores = score_pool[l_idx]

            score_pool[l_idx]['costs'] = scores['costs'] * size

        return score_pool


    def _compute_gradient(self, model, input_neg, desc=True):
        """Compute gradient.

        Arachne sorts the neural weights according to the gradient loss
        of the faulty input backpropagated to the corresponding neuron.

        :param model:
        :param input_neg:
        :param desc:
        :return:
        """
        # For return
        candidates = []

        #res=run_localise.get_target_weights(model,None,None)
        #print(res)

        # Identify class of loss function
        loss_func = keras.losses.get(model.loss)
        layer_index = len(model.layers) - 1
        layer = model.get_layer(index=layer_index)

        # Evaluate grad on neural weights
        with tf.GradientTape() as tape:
            # import pdb; pdb.set_trace()
            logits = model(input_neg[0])  # get the forward pass gradient
            loss_value = loss_func(input_neg[1], logits)
            grad_kernel = tape.gradient(
                loss_value, layer.kernel
            )  # TODO bias?# Evaluate grad on neural weights

        for j in trange(grad_kernel.shape[1], desc="Computing gradient"):
            for i in range(grad_kernel.shape[0]):
                dl_dw = grad_kernel[i][j]
                # Append data tuple
                # (layer, i, j) is for identifying neural weight
                candidates.append([layer_index, i, j, np.abs(dl_dw)])

        # Here we do not need to sort candidates in order of grad loss
        # candidates.sort(key=lambda tup: tup[3], reverse=desc)

        return candidates

    def local_GradientLoss(self,test_mode=True):



        # "N_g is set to be the number of negative inputs to repair
        # multiplied by 20"
        if self.num_grad is None:
            self.num_grad = len(self.input_neg[0]) * 20

        img_shape  = self.input_neg[0].shape
        label_shape = self.input_neg[1].shape

        self.input_neg[0] = np.reshape(self.input_neg[0],(img_shape[0]*img_shape[1],img_shape[2],img_shape[3],img_shape[4]))
        self.input_neg[1] = np.reshape(self.input_neg[1], (label_shape[0] * label_shape[1], label_shape[2]))

        img_shape = self.input_pos[0].shape
        label_shape = self.input_pos[1].shape

        self.input_pos[0] = np.reshape(self.input_pos[0],(img_shape[0] * img_shape[1], img_shape[2], img_shape[3], img_shape[4]))
        self.input_pos[1] = np.reshape(self.input_pos[1], (label_shape[0] * label_shape[1], label_shape[2]))

        reshaped_model = self._reshape_target_model(self.model, self.input_neg)

        # First compute scores on Neg inputs
        candidates_neg = self._compute_gradient(reshaped_model, self.input_neg)

        for i in range(len(candidates_neg)):
            candidates_neg[i][3] *= self.neg_size

        # Now on Pos inputs
        candidates_pos = self._compute_gradient(reshaped_model, self.input_pos)

        for i in range(len(candidates_pos)):
            candidates_pos[i][3] *= self.pos_size

        #self.model = reshaped_model


        return candidates_neg, candidates_pos

    def local_Activation(self):

        """
        Computes the sum of activations for the computation of forward impact. For each
        candidate weight we compute the average activation of the previous neuron.
        Returns
        -------

        """

        layer_index = self.target_layer-1
        previous_layer = self.model.get_layer(index=layer_index)

        # Evaluate activation value of the corresponding neuron
        # in the previous layer
        get_activations = K.function([self.model.input], previous_layer.output)

        print("Computing activations...",end="")
        activations_neg = get_activations(self.input_neg[0])
        activations_pos = get_activations(self.input_pos[0])
        print("DONE")

        activations_neg = [sum(i) for i in zip(*activations_neg)]
        activations_pos = [sum(i) for i in zip(*activations_pos)]

        activations_neg = [x * self.neg_size for x in activations_neg]
        activations_pos = [x * self.neg_size for x in activations_pos]


        return activations_neg, activations_pos

    def local_OutputGrad(self):
        """
        Computes the gradient of model's output w.r.t. activation of one neuron
        Returns
        -------

        """

        # Modify model to return activation of interest

        with tf.GradientTape(persistent=True) as tape:
            layer_index = self.target_layer#-3
            curr_layer = self.model.get_layer(index=layer_index)

            new_model = tf.keras.Model(inputs=self.model.input,outputs=[self.model.output,curr_layer.output])

            output_neg = new_model(self.input_neg[0])
            output_pos = new_model(self.input_pos[0])

            #get_activations = K.function([self.model.input], curr_layer.output)
            #
            #activ_neg = get_activations(tf.convert_to_tensor(self.input_neg[0]))
            #activ_pos = get_activations(self.input_pos[0])

        # These lists contain the output gradient for input from the dataset.
        # Since we want the aggregated output gradient, we compute the sum locally
        grad_neg = tape.gradient(output_neg[0],output_neg[1])
        grad_pos = tape.gradient(output_pos[0], output_pos[1])

        num_grads = grad_neg.shape[1]

        sum_grad_neg = [0]*num_grads
        sum_grad_pos = [0]*num_grads


        for i in range(num_grads):

            s =0

            for j in range(self.neg_size):

                s+=grad_neg[j][i]
            sum_grad_neg[i]=s

            s = 0

            for j in range(self.pos_size):
                s += grad_pos[j][i]
            sum_grad_pos[i] = s

        return sum_grad_neg,sum_grad_pos

    def save_FL(self,grad_loss,act=None,grad_output=None,data_type='Negative',folder="results",output_dir=""):
        """
        Saves the intermediate results of FL as numpy arrays.
        Parameters
        ----------
        grad_loss
        act
        grad_output

        Returns
        -------

        """

        path = Path(output_dir)

        if not os.path.exists(path):
            os.mkdir(path)

        path = path / Path(f"{data_type}_{folder}")

        if not os.path.exists(path):
            os.mkdir(path)

        client_path = path / f"Client_{self.id}"

        if not os.path.exists(client_path):
            os.mkdir(client_path)

        # Save client's GL
        grad_loss_path = client_path /"GL"
        if not os.path.exists(grad_loss_path):
            os.mkdir(grad_loss_path)

        for l_idx in grad_loss:
            layer_path = grad_loss_path / f"layer_{l_idx}"
            scores = grad_loss[l_idx]['costs']
            np.save(layer_path,scores)

        # Save client's Activation
        act_path = client_path/ "Act"
        if not os.path.exists(act_path):
            os.mkdir(act_path)

        for l_idx in act:
            layer_path = act_path / f"layer_{l_idx}"
            scores = act[l_idx]['costs']
            np.save(layer_path,scores)

        # Save client's gradient to Output
        out_path = client_path / "Out"
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        for l_idx in grad_output:
            layer_path = out_path /  f"layer_{l_idx}"
            scores = grad_output[l_idx]['costs']
            np.save(layer_path,scores)


    def load_FL_pos(self,path,type="GL"):
        """
        Loads FL scores computed on positive inputs
        Parameters
        ----------
        path

        Returns
        -------

        """

        assert type in ['GL','Act','Out'], f"Unsupported type of FL score to load"

        res = {}

        path = Path(path)/Path(type)

        for file_path in os.listdir(path):

            #print(os.path.isfile(file_path) , ("layer_" in file_path))
            if ("layer_" in file_path):
                layer_num = file_path.split("_")[1].split(".")[0]
                layer_num = int(layer_num)

                data = np.load(path/Path(file_path),allow_pickle=True)
                res[layer_num] = {}
                res[layer_num]['shape'] = data.shape
                res[layer_num]['costs'] = data

        return res


#######################################################

# ***Methods for Fault Localisation using Arachnev2 ***

#######################################################

    def local_FI_and_GL(self,output_dir="outputs/VGG16",target_layer_list=[],recompute=True):

        t_begin=time.time()

        # Get the target layers
        target_layers = get_target_weights(None,self.model_dir)
        #target_layers.pop(1)
        #target_layers={13:target_layers[13]}
        target_layers = {l:target_layers[l] for l in target_layer_list}

        tf.compat.v1.disable_eager_execution()

        # First compute FI and GL for neg input
        X = self.input_neg[0]
        y = self.input_neg[1]
        indices_to_neg = list(range( self.neg_size ))
        res_neg, act_neg, grad_neg = compute_FI_and_GL( X,y,
                                                        indices_to_target=indices_to_neg,
                                                        target_weights=target_layers,
                                                        is_multi_label=True,
                                                        path_to_keras_model=self.model_dir,
                                                        federated=True)

        res_neg = self.rescale_scores(res_neg,True)
        act_neg = self.rescale_scores(act_neg,True)
        grad_neg = self.rescale_scores(grad_neg,True)



        self.save_FL(res_neg,grad_output=grad_neg,act=act_neg,data_type='Negative',output_dir=output_dir)

        # Then compute FI and GL for pos input
        X = self.input_pos[0]
        y = self.input_pos[1]
        indices_to_pos = list(range(0, len(y)))

        # Perform the computation only if it was not done before
        pos_fl_path = Path(f"{output_dir}/Positive_results/Client_{self.id}")
        if recompute or (not os.path.exists(pos_fl_path)):
            print(f"File {pos_fl_path} not found. Going for first computation")
            res_pos, act_pos, grad_pos = compute_FI_and_GL(X, y,
                                                                        indices_to_target=indices_to_pos,
                                                                        target_weights=target_layers,
                                                                        is_multi_label=True,
                                                                        path_to_keras_model=self.model_dir,
                                                                        federated=True)
        else:
            print(f"Loaded positive FL scores. Skipping computation.")
            res_pos = self.load_FL_pos(pos_fl_path,"GL")
            act_pos = self.load_FL_pos(pos_fl_path,"Act")
            grad_pos = self.load_FL_pos(pos_fl_path,"Out")

        res_pos = self.rescale_scores(res_pos, True)
        act_pos = self.rescale_scores(act_pos, True)
        grad_pos = self.rescale_scores(grad_pos, True)

        self.save_FL(res_pos,grad_output=grad_pos,act=act_pos, data_type='Positive',output_dir=output_dir)

        # Report the computed metrics
        for tl_idx in res_neg:
            print("LAYER", tl_idx)
            print("On negative inputs:")
            print(f"\tGRAD LOSS: shape {res_neg[tl_idx]['shape']}")
            print(f"\tAVERAGED ACTIVATIONS: shape {act_neg[tl_idx]['shape']}")
            print(f"\tGRAD OUTPUT: shape {grad_neg[tl_idx]['shape']}")
            print("On positive inputs:")
            print(f"\tGRAD LOSS: shape {res_pos[tl_idx]['shape']}")
            print(f"\tAVERAGED ACTIVATIONS: shape {act_pos[tl_idx]['shape']}")
            print(f"\tGRAD OUTPUT: shape {grad_pos[tl_idx]['shape']}")

        t_end=time.time()

        diff = (t_end - t_begin)*1000

        print(f"Client {self.id} took {diff} milliseconds to localise")

        # Log the message exchanged
        path = Path(output_dir)/Path("message_log.txt")
        msg_size = getsizeof(res_neg) + getsizeof(res_pos) + getsizeof(act_neg) + getsizeof(act_pos) +  getsizeof(grad_neg) + getsizeof(grad_pos)
        f = open(path,"a")
        f.write(f"[{time.time()}] Client {self.id} sends {msg_size} bytes after local FL\n")
        f.close()


        return res_neg, act_neg, grad_neg, res_pos, act_pos, grad_pos

    def fitness(self):
        """
        Computes arachnev2 fitness on local datasets
        Returns
        -------

        """

        img_neg = self.input_neg[0]
        label_neg = self.input_neg[1]
        score_neg = 0

        predictions = self.model.predict(img_neg)

        for i in range(len(img_neg)):

            if label_neg[i] == predictions[i]:
                score_neg+=1
            else:
                loss = self.model.compute_loss(x=img_neg[i],y=label_neg[i])
                score_neg += 1/(1+loss)

        img_pos = self.input_pos[0]
        label_pos = self.input_pos[1]
        score_pos = 0

        predictions = self.model.predict(img_pos)

        for i in range(len(img_pos)):

            if label_neg[i] == predictions[i]:
                score_pos += 1
            else:
                loss = self.model.compute_loss(x=img_pos[i], y=label_pos[i])
                score_pos += 1 / (1 + loss)

        return score_neg + self.alpha*score_pos





