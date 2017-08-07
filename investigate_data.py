import numpy as np
    
data = np.load('energies.npz')

data2 = np.load('energy_seq.npz')
    
data3 = np.load('energy_seq_pssm.npz')
# ['id', 'x', 'split', 'y', 'mask_seq', 'E_names']
    
# Access data - data['y']

# The archives contain the following arrays:
#- y: Energy targets (nseq x seqlen x nterm)
#- x or aa: Predictor (nseq x seqlen x nvar ) where the variables can be one-hot
# encoded amino acids and PSSM values
#- E_names: Names of energy terms (targets)
#- id: Identifier for each protein (pdb id and chain name)
#- split: Cross validation assignments for splitting data randomly 
#  but consistently)
#- mask_seq: (nseq x seqlen x 1) Mask marking the parts of each sequence that
#  is actual sequence (not padding).
#print (data.files)
#print (data2.files)
print (data3.files)
print (data3['E_names'])

all_aminoacids = (data3['x'])
all_energies = (data3['y'])
all_masks = (data3['mask_seq'])

print (all_aminoacids.shape)
print (all_energies.shape)

#all_data = np.concatenate(all_aminoacids, all_masks)
print(all_masks.shape)
#print (all_ids[0])
print (all_aminoacids[0].shape)
#print (all_splits[0])
#print (all_energies[0])
#print (all_sequences[0])
pssm = all_aminoacids[1]
print(pssm)

