from keras.models import Model
from utils.layers import Conv1D, _activation
from keras.layers import Input, Concatenate, Reshape, Dense, Add,\
                         Activation, Flatten, Dropout, BatchNormalization,\
                         Flatten, RepeatVector, LocallyConnected1D, ZeroPadding1D,\
                         GRU
from keras.layers.pooling import GlobalAveragePooling1D


def fc_encoder(seqlen, latent_dim, alphabet_size=21, encoder_hidden=[250,250,250],
               encoder_dropout=[0.7,0.,0.], activation='relu', n_conditions=0):
    
    x = Input(shape=(seqlen, alphabet_size,))
    h = Flatten()(x)
    
    if n_conditions>0:
        conditions = Input((n_conditions,))
        h = Concatenate()([h, conditions])
    
    for n_hid, drop in zip(encoder_hidden, encoder_dropout):
        h = Dense(n_hid, activation=activation)(h)
        if drop > 0:
            h = Dropout(drop)(h)

    #Variational parameters
    z_mean=Dense(latent_dim)(h)
    z_var=Dense(latent_dim, activation='softplus')(h)
    
    if n_conditions > 0:
        E = Model([x, conditions], [z_mean, z_var])
    else:
        E = Model(x, [z_mean, z_var])
    return E

def cnn_encoder(original_dim, latent_dim,
                nchar=21, num_filters=21, kernel_size=2,
                BN=True, activation='prelu', dropout=None,
                log_transform_var=False, max_filters=10000, n_conv=5,
                n_conditions=None, n_dense_cond=6, cond_concat_dim=None):

    x = Input((original_dim, nchar))
    h = x
    for i in range(n_conv):
        h = Conv1D(min(num_filters*(2**i), max_filters), kernel_size, activation=activation,
                   strides=1 if i==0 else 2,
                   use_bias=not BN, BN=BN, dropout=dropout)(h)
    
    h = Flatten()(h)

    if n_conditions>0:
        conditions, h_cond = cond_mlp(h._keras_shape[-1] if cond_concat_dim is None else cond_concat_dim,
                                      activation=activation, n_conditions=n_conditions, h=n_dense_cond)
        h = Concatenate()([h, h_cond])
    
    z_mean = Dense(latent_dim)(h)
    z_var = Dense(latent_dim, activation='softplus')(h)

    if n_conditions>0:
        E = Model([x, conditions], [z_mean, z_var])
    else:
        E = Model(x, [z_mean, z_var])
    return E

def cond_mlp(out_dim, n_layers=2, h=6, n_conditions=3,
             activation='prelu'):
    conditions = Input((n_conditions,))
    h_cond = conditions
    for i in range(n_layers):
        h_cond = _activation(activation, BN=False)(Dense(h)(h_cond))

    h_cond = Dense(out_dim)(h_cond)
    h_cond = _activation(activation, BN=False)(h_cond)

    return conditions, h_cond