import tensorflow as tf

from functools import partial
import numpy as np


from keras import backend as K
from keras import objectives, losses
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, Callback
from keras.models import Model
from keras.layers import Input, Lambda

from utils import aa_letters, luxa_seq
from utils.metrics import aa_acc
from utils.data_loaders import right_pad, to_one_hot
from utils.decoding import _decode_ar, _decode_nonar, batch_temp_sample

nchar = len(aa_letters)  # = 21

def sampler(latent_dim, epsilon_std=1):
    _sampling = lambda z_args: (z_args[0] + K.sqrt(tf.convert_to_tensor(z_args[1] + 1e-8, np.float32)) *
                                K.random_normal(shape=K.shape(z_args[0]), mean=0., stddev=epsilon_std))

    return Lambda(_sampling, output_shape=(latent_dim,))

def luxa_batch_conds(n_samples, solubility_level):
    target_conds = [0,0,0]
    target_conds[solubility_level] = 1
    target_conds = np.repeat(np.array(target_conds).reshape((1,3)), n_samples, axis=0)
    return target_conds


class BaseProtVAE:
    # Child classes must define a self.E, self.G
    def __init__(self, n_conditions=0, autoregressive=True,
                 lr=0.001, clipnorm=0., clipvalue=0., metrics=['accuracy', aa_acc],
                 condition_encoder=True, latent_dim=50, original_dim=504):

        self.n_conditions = n_conditions
        self.condition_encoder = condition_encoder
        self.autoregressive = autoregressive
        self.latent_dim = latent_dim
        self.original_dim = original_dim

        self.S = sampler(latent_dim, epsilon_std=1.)

        prot = self.E.inputs[0]
        encoder_inp = [prot]
        vae_inp = [prot]
        
        if n_conditions>0:
            conditions = Input((n_conditions,))
            vae_inp.append(conditions)
            if condition_encoder:
                encoder_inp.append(conditions)

        z_mean, z_var = self.E(encoder_inp)
        z = self.S([z_mean, z_var])  # sampler
        self.stochastic_E = Model(inputs=encoder_inp, outputs=[z_mean, z_var, z])
        
        decoder_inp = [z]
        if n_conditions > 0:
            decoder_inp.append(conditions)

        if autoregressive:
            decoder_inp.append(prot)

        decoded = self.G(decoder_inp)
        
        self.VAE = Model(inputs=vae_inp, outputs=decoded)

        def xent_loss(x, x_d_m):
            return K.sum(objectives.categorical_crossentropy(x, x_d_m), -1)

        def kl_loss(x, x_d_m):
            return - 0.5 * K.sum(1 + K.log(z_var + 1e-8) - K.square(z_mean) - z_var, axis=-1)

        def vae_loss(x, x_d_m):
            return xent_loss(x, x_d_m) + kl_loss(x, x_d_m)

        log_metrics = metrics + [xent_loss, kl_loss, vae_loss]

        print('Learning rate ', lr)
        self.VAE.compile(loss=vae_loss, optimizer=Adam(lr=lr, clipnorm=clipnorm, clipvalue=clipvalue),
                         metrics=log_metrics)
        self.metric_names = metric_names = ['loss'] + [m.__name__ if type(m)!=str else m for m in self.VAE.metrics ]
        print('Protein VAE initialized !')

    def load_weights(self, file='generative_models/weights/default.h5'):
        self.VAE.load_weights(file)
        print('Weights loaded !')
        return self

    def save_weights(self, file='generative_models/weights/default.h5'):
        self.VAE.save_weights(file)
        print('Weights saved !')
        return self

    def prior_sample(self, n_samples=1, mean=0, stddev=1,
                     remove_gaps=False, batch_size=5000):
        if n_samples > batch_size:
            x = []
            total = 0
            while total< n_samples:
                this_batch = min(batch_size, n_samples - total)
                z_sample = mean + stddev * np.random.randn(this_batch, self.latent_dim)
                x += self.decode(z_sample, remove_gaps=remove_gaps)
                total += this_batch
        else:
            z_sample = mean + stddev * np.random.randn(n_samples, self.latent_dim)
            x = self.decode(z_sample, remove_gaps=remove_gaps)
        return x

    def decode(self, z, remove_gaps=False, sample_func=None, conditions=None):
        if self.autoregressive:
            return _decode_ar(self.G, z, remove_gaps=remove_gaps, sample_func=sample_func,
                              conditions=conditions)
        else:
            return _decode_nonar(self.G, z, remove_gaps=remove_gaps, conditions=conditions)

    def generate_variants_luxA(self, num_samples, posterior_var_scale=1., temperature=0.,
                               solubility_level=None):

        luxa_oh = to_one_hot(right_pad([luxa_seq], self.E.input_shape[1]))
        luxa_oh = np.repeat(luxa_oh, num_samples, axis=0)
        orig_conds = np.repeat(np.array([1,0,0]).reshape((1,3)), num_samples, axis=0)
        inputs = luxa_oh if solubility_level is None else [luxa_oh, orig_conds]

        luxa_zmean, luxa_zvar, luxa_z = self.stochastic_E.predict(inputs)
        print(luxa_z.shape)

        if posterior_var_scale != 1.:
            luxa_z = np.sqrt(posterior_scale*luxa_zvar)*np.random.randn(*luxa_zmean.shape) + luxa_zmean

        sample_func = None
        if temperature > 0:
            sample_func = partial(batch_temp_sample, temperature=temperature)
        target_conds = None if solubility_level is None else luxa_batch_conds(num_samples, solubility_level)
        return self.decode(luxa_z, remove_gaps=True, sample_func=sample_func,
                           conditions=target_conds)