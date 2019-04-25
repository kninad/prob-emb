
from hype.graph import eval_reconstruction, load_adjacency_matrix
import argparse
import numpy as np
import torch
import os
from hype.lorentz import LorentzManifold
from hype.euclidean import EuclideanManifold, TranseManifold
from hype.poincare import PoincareManifold
import pickle

def get_model_embeddings(fname):
    chkpnt = torch.load(fname)
    dset = chkpnt['conf']['dset']
    if not os.path.exists(dset):
        raise ValueError("Can't find dset!")
    data_format = 'hdf5' if dset.endswith('.h5') else 'csv'
    #dset = load_adjacency_matrix(dset, 'hdf5')
    dset = load_adjacency_matrix(dset, data_format)
    
    info_dict = {}    
#    info_dict['dset'] = dset
    info_dict['idmap'] = dset['idmap']
    info_dict['embeddings'] = chkpnt['embeddings'].numpy()
    return info_dict

def get_vector(entity_id, info_dict):
    imap = info_dict['idmap']
    idx_obj = imap[entity_id]
    vec = info_dict['embeddings'][idx_obj, :]
    return vec

def distance(u, v, eps=1e-5):
    # Calculate the distance between two vectors in poincare space
    squnorm = np.clip(np.sum(u * u, axis=-1), 0, 1 - eps)
    sqvnorm = np.clip(np.sum(v * v, axis=-1), 0, 1 - eps)
    sqdist = np.sum(np.power(u - v, 2), axis=-1)
    x = 2 * sqdist / ((1 - squnorm) * (1 - sqvnorm)) + 1
    # arcosh
    tmp = np.sqrt(np.power(z, 2) - 1)
    return np.log(x + tmp)



def main():    
    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Process the model chkpnt path')
    parser.add_argument('file', help='Path to checkpoint')
    args = parser.parse_args()
    
    model_file = args.file
    info_dict = get_model_embeddings(model_file)
    

if __name__ == '__main__':
    main()
