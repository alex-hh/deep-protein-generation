import numpy as np
# from generative_models.rnn_lm import rnn_lm
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Concatenate, Reshape, Dense, Add,Lambda,\
                         Activation, Flatten, Dropout, BatchNormalization,\
                         Flatten, RepeatVector, ZeroPadding1D,\
                         GRU, LSTM, Conv1D
from utils.layers import _activation, Deconv1D



def fc_decoder(latent_dim, seqlen, decoder_hidden=[250], decoder_dropout=[0.],
               alphabet_size=21, n_conditions=0, activation='relu'):
    latent_vector = Input((latent_dim,))
    latent_v = latent_vector
    if n_conditions > 0:
        conditions = Input((n_conditions,))
        latent_v = Concatenate()([latent_v, conditions])
    
    decoder_x = latent_v
    
    for h, d in zip(decoder_hidden, decoder_dropout):
        decoder_x = Dense(h, activation=activation)(decoder_x)
        if n_conditions > 0:
            decoder_x = Concatenate()([decoder_x, conditions])

        decoder_x = Dropout(d)(decoder_x)

    decoder_out = Dense(seqlen*alphabet_size, activation=None)(decoder_x)
    decoder_out = Reshape((seqlen, alphabet_size))(decoder_out)
    decoder_out = Activation('softmax')(decoder_out)

    if n_conditions > 0:
        G = Model([latent_vector, conditions], decoder_out)
    else:
        G = Model(latent_vector, decoder_out)

    return G

def upsampler(latent_vector, low_res_dim, min_deconv_dim=21,
              n_deconv=3, kernel_size=2, BN=True, dropout=None,
              activation='relu', max_filters=336):

    low_res_features = min(min_deconv_dim * (2**n_deconv), max_filters)
    h = Dense(low_res_dim * low_res_features, name='upsampler_mlp')(latent_vector)
    h = Reshape((low_res_dim, low_res_features))(h)
    
    for i in range(n_deconv):
        h = Deconv1D(min(min_deconv_dim * 2**(n_deconv-(i+1)), max_filters), kernel_size,
                     strides=2, activation=activation,
                     use_bias=not BN, BN=BN, dropout=dropout)(h)
    return h

def recurrent_sequence_decoder(latent_dim, seqlen, ncell=512,
                               alphabet_size=21, project_x=True,
                               upsample=False, min_deconv_dim=42,
                               input_dropout=None, intermediate_dim=63,
                               max_filters=336, n_conditions=0,
                               cond_concat_each_timestep=False):

    latent_vector = Input((latent_dim,))
    latent_v = latent_vector

    prot_oh = Input((seqlen, alphabet_size))
    input_x = ZeroPadding1D(padding=(1,0))(prot_oh)
    input_x = Lambda(lambda x_: x_[:,:-1,:])(input_x)
    
    if input_dropout is not None:
        input_x = Dropout(input_dropout, noise_shape=(None, seqlen, 1))(input_x)
    if project_x:
        input_x = Conv1D(alphabet_size, 1, activation=None, name='decoder_x_embed')(input_x)

    if n_conditions > 0:
        cond_inp = Input((n_conditions,))
        conditions = cond_inp
        latent_v = Concatenate()([latent_v, conditions])
    
    rnn = GRU(ncell, return_sequences=True)
    if upsample:
        z_seq = upsampler(latent_v, intermediate_dim, min_deconv_dim=min_deconv_dim,
                          n_deconv=3, activation='prelu', max_filters=max_filters)
        if cond_concat_each_timestep:
            cond_seq = RepeatVector(seqlen)(conditions)
            z_seq = Concatenate(axis=-1)([z_seq, cond_seq])
    else:
        z_seq = RepeatVector(seqlen)(latent_v)
    
    xz_seq = Concatenate(axis=-1)([z_seq, input_x])
    rnn_out = rnn(xz_seq)
    
    processed_x = Conv1D(alphabet_size, 1, activation=None, use_bias=True)(rnn_out)
    output = Activation('softmax')(processed_x)
    
    if n_conditions > 0:
        G = Model([latent_vector, cond_inp, prot_oh], output)
    else:
        G = Model([latent_vector, prot_oh], output)
    return G
    