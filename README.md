# Federated Repair of Deep Neural Networks
As Deep Neural Networks (DNNs) are embedded in more and more critical systems, ensuring they perform well on some specific inputs is essential. DNN repair techniques have shown good results when fixing specific misclassifications in already trained models using additional data, even surpassing additional training. In safety-critical applications, such as autonomous driving perception, collaboration between industrial actors would lead to more representative datasets for repair, that would enable to obtain more robust models and thus safer systems. These companies, however, are often reluctant to share their data, to both protect their intellectual property and the privacy of their users. Federated Learning is an approach that allows for collaborative, privacy-preserving training of DNNs. Inspired by this technique, this work proposes *Federated Repair* in order to collaboratively repair a DNN model without the need for sharing any raw data. We implemented Federated Repair based on a state-of-the-art DNN repair technique, and applied it to three DNN models, with federation size from 2 to 10. Results show that Federated Repair can achieve the same repair efficiency as non-federated DNN repair using the pooled data, despite the presence of rounding errors when aggregating clients' results.

# How to run

## Install virtual environment

Create and install a dedicated virtual environment to run the experiments. You can use the following commands:

```
$ python -m venv <path/to/venv>
$ source <path/to/venv>/bin/activate
(venv-name) $ pip install --upgrade pip
(venv-name) $ pip install -e.
```

## Preparing the data

You need a DNN model and a repair dataset  `repair.h5`. The code supports DNNs saved in  `saved_model.pb` and `keras_metadata.pb`. Place all of them in the same folder, then run
 ```
 repair target --model_dir=<path_to_folder> --target_dir=<path_to_folder>
```
 This generates two subfolders  `<path_to_folder>/negative` (negative inputs) and  `<path_to_folder>/negative`

## Running the experiments
 To run an experiment, type and run command
 ```
 ./run_exp.sh <class_number> <num_clients> <DNN> <GPU_number>
```
 This starts a federated repair on model <DNN> using <num_clients> clients on GPU <GPU_number>. The repair will fix misclassifications from <class_number> to anything. The script runs federated FL followed by federated DE.

This repo already provides the experimental results for DNNs VGG16, VGG19, and EnetB0 (ad described in our paper). FL results are in `fedrep_results_FL.zip`, while DE results are in `fedrep_results_DE.zip`.

## People
* Davide Li Calsi https://www.linkedin.com/in/davide-li-calsi-4a4968206/
* Thomas Laurent https://laurenttho3.github.io
* Paolo Arcaini http://group-mmm.org/~arcaini/
* Fuyuki Ishikawa http://research.nii.ac.jp/~f-ishikawa/en/

## Paper
D. Li Calsi, T. Laurent, P. Arcaini, F. Ishikawa. Federated Repair of Deep Neural Networks. In DeepTest 2024, Lisbon, Portugal, April 20, 2024
