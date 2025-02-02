import argparse
import numpy as np
from ARCUS import ARCUS
from datasets.data_utils import load_dataset, load_datasets_spectra
from utils import set_gpu, set_seed
import pandas as pd

parser = argparse.ArgumentParser()

# Define dataset path and name as default values
default_path_dataset = r'datasets/ARCUS_DS/'  # Default dataset path

default_layer_num = 3
default_learning_rate = 0.0001


"""The hidden dim size of AE. \
Manually chosen or the number of pricipal component explaining at least 70% of variance: \
MNIST_AbrRec: 24,  MNIST_GrdRec: 25, F_MNIST_AbrRec: 9, F_MNIST_GrdRec: 15, GAS: 2, RIALTO: 2, INSECTS_Abr: 6, \
INSECTS_Incr: 7, INSECTS_IncrGrd: 8, INSECTS_IncrRecr: 7"""
default_name_dataset = 'RIALTO'  # Default dataset name
default_hidden_dim = 2

parser.add_argument('--model_type', type=str, default="RAPP", choices=["RAPP", "RSRAE", "DAGMM"])
parser.add_argument('--inf_type', type=str, default="ADP", choices=["ADP", "INC"], help='INC: drift-unaware, ADP: drift-aware')

parser.add_argument('--dataset_path', type=str, default=default_path_dataset, help="Path to dataset directory")
parser.add_argument('--dataset_name', type=str, default=default_name_dataset, choices=["MNIST_AbrRec", "MNIST_GrdRec", "F_MNIST_AbrRec",
                                                                                       "F_MNIST_GrdRec", "GAS", "RIALTO", "INSECTS_Abr",
                                                                                       "INSECTS_Incr", "INSECTS_IncrGrd", "INSECTS_IncrRecr"],
                    help="Name of the dataset")

parser.add_argument('--seed', type=int, default=42, help="random seed")
parser.add_argument('--gpu', type=str, default='0', help="foramt is '0,1,2,3'")
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--min_batch_size', type=int, default=32)
parser.add_argument('--init_epoch', type=int, default=5)
parser.add_argument('--intm_epoch', type=int, default=1)
parser.add_argument('--hidden_dim', type=int, default=default_hidden_dim, \
                    help="The hidden dim size of AE. \
                          Manually chosen or the number of pricipal component explaining at least 70% of variance: \
                          MNIST_AbrRec: 24,  MNIST_GrdRec: 25, F_MNIST_AbrRec: 9, F_MNIST_GrdRec: 15, GAS: 2, RIALTO: 2, INSECTS_Abr: 6, \
                          INSECTS_Incr: 7, INSECTS_IncrGrd: 8, INSECTS_IncrRecr: 7")
parser.add_argument('--layer_num', type=int, default=default_layer_num, help="Num of AE layers")
parser.add_argument('--RSRAE_hidden_layer_size', type=int, nargs="+", default=[32, 64, 128], \
                    help="Suggested by the RSRAE author. The one or two layers of them may be used according to data sets")
parser.add_argument('--learning_rate', type=float, default=default_learning_rate)
parser.add_argument('--reliability_thred', type=float, default=0.95, help='Threshold for model pool adaptation')
parser.add_argument('--similarity_thred', type=float, default=0.80, help='Threshold for model merging')

args = parser.parse_args()
args = set_gpu(args)
set_seed(args.seed)

#args, loader = load_dataset(args)

# Paths
path_spectral_data = r"C:\Users\nicol\ARCUS\datasets\RecordsALPS\RecordsALPS_V3 - copia"
path_ground_thruth = r"C:\Users\nicol\ARCUS\datasets\RecordsALPS\GroundTruth\V3"


loader_lists, input_dims, dataset_names = load_datasets_spectra(args, path_spectral_data, path_ground_thruth)
print("len loader:", len(loader_lists))

df_results = pd.DataFrame(columns=['dataset_namet', 'auc_result'])
for i in range(len(loader_lists)):
    loader = loader_lists[i]
    args.input_dim = input_dims[i]
    args.dataset_name = dataset_names[i]

    print("i: ", i)
    print("loader: ", loader)
   

    ARCUS_instance = ARCUS(args)
    returned, auc_hist, anomaly_scores = ARCUS_instance.simulator(loader)

    if(returned):
        print("Data set: ",args.dataset_name)
        
        print("Model type: ", args.model_type)
        auc = np.round(np.mean(auc_hist),3)
        print("AUC: ", auc)
        
        # Append the new row
        new_row = {
            'dataset_namet': args.dataset_name,
            'auc_result': auc,
        }
        new_row = pd.DataFrame([new_row])
        df_results = pd.concat([df_results, new_row], ignore_index=True)
    else:
        print("Error occurred")

print("Results: ", df_results)