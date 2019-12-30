from utils import aa_letters
from models.protcnn import BaseProtVAE, sampler
from models.encoders import cnn_encoder, fc_encoder
from models.decoders import recurrent_sequence_decoder, fc_decoder


class MSAVAE(BaseProtVAE):
    def __init__(self, latent_dim=10, original_dim=360,
                 n_conditions=0., activation='relu',
                 encoder_kwargs={'encoder_hidden': [256, 256],
                                 'encoder_dropout': [0., 0.]},
                 decoder_kwargs={'decoder_hidden': [256,256],
                                 'decoder_dropout': [0.,0.]}):
        self.E = fc_encoder(original_dim, latent_dim,
                            n_conditions=n_conditions,
                            activation=activation,
                            **encoder_kwargs)
        self.G = fc_decoder(latent_dim, original_dim,
                            n_conditions=n_conditions,
                            activation=activation,
                            **decoder_kwargs)
        super().__init__(latent_dim=latent_dim, original_dim=original_dim,
                         n_conditions=n_conditions, autoregressive=False)

class ARVAE(BaseProtVAE):
    def __init__(self, original_dim=504, latent_dim=50,
                 clipnorm=5, lr=0.001, n_conditions=0,
                 encoder_kwargs={'num_filters': 21, 'kernel_size': 2},
                 decoder_kwargs={'upsample': True, 'ncell': 512, 'input_dropout': 0.45}):
        self.E = cnn_encoder(original_dim, latent_dim,
                             n_conditions=n_conditions,
                             **encoder_kwargs)
        self.G = recurrent_sequence_decoder(latent_dim, original_dim,
                                            n_conditions=n_conditions,
                                            **decoder_kwargs)
        super().__init__(latent_dim=latent_dim, original_dim=original_dim,
                         autoregressive=True, clipnorm=clipnorm, lr=lr,
                         n_conditions=n_conditions)