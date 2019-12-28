import numpy as np

from utils import aa_letters


def to_string(seqmat, remove_gaps=True):
    a = [''.join([aa_letters[np.argmax(aa)] for aa in seq]) for seq in seqmat]
    return [x.replace('-', '') for x in a] if remove_gaps else a

def greedy_decode_1d(arr1d):
    a = np.zeros(arr1d.shape)
    i = np.argmax(arr1d)
    a[i] = 1
    return a

def greedy_decode(pred_mat):
    return np.apply_along_axis(greedy_decode_1d, -1, pred_mat)

def _decode_nonar(generator, z, remove_gaps=False, alphabet_size=21, conditions=None):
    xp = generator.predict(z) if conditions is None else generator.predict([z, conditions])
    x = greedy_decode(xp)
    return to_string(x, remove_gaps=remove_gaps)

def _decode_ar(generator, z, remove_gaps=False, alphabet_size=21,
               sample_func=None, conditions=None):
    original_dim, alphabet_size = generator.output_shape[1], generator.output_shape[-1]
    x = np.zeros((z.shape[0], original_dim, alphabet_size))
    start = 0
    for i in range(start, original_dim):
        # iteration is over positions in sequence, which can't be parallelized
        pred = generator.predict([z, x]) if conditions is None else generator.predict([z, conditions, x])
        pos_pred = pred[:, i, :]
        if sample_func is None:
            pred_ind = pos_pred.argmax(-1) # convert probability to index
        else:
            pred_ind = sample_func(pos_pred)
        for j, p in enumerate(pred_ind):
            x[j, i, p] = 1

    seqs = to_string(x, remove_gaps=remove_gaps)
    return seqs

def batch_temp_sample(preds, temperature=1.0):
    batch_sampled_aas = []
    for s in preds:
        batch_sampled_aas.append(temp_sample_outputs(s, temperature=temperature))
    out = np.array(batch_sampled_aas)
    return out

def temp_sample_outputs(preds, temperature=1.0):
    # https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
    # helper function to sample an index from a probability array N.B. this works on single prob array (i.e. array for one sequence at one timestep)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)