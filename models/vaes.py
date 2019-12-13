from utils import aa_letters
from models.protcnn import BaseProtVAE, sampler
from models.encoders import cnn_encoder, fc_encoder
from models.decoders import recurrent_sequence_decoder, fc_decoder


class MSAVAE(BaseProtVAE):
  def __init__(self, latent_dim=10, original_dim=504,
               alphabet=aa_letters, activation='relu', encoder_hidden=[256,256],
               encoder_dropout=[0.,0.], decoder_hidden=[256,256],
               decoder_dropout=[0.,0.], n_conditions=0.):
      self.latent_dim = latent_dim
      self.original_dim = original_dim
      self.E = fc_encoder(original_dim, len(alphabet), latent_dim,
                          encoder_hidden=encoder_hidden, activation=activation,
                          encoder_dropout=encoder_dropout, n_conditions=n_conditions)
      self.S = sampler(latent_dim, epsilon_std=1.)
      self.G = fc_decoder(latent_dim, original_dim, decoder_hidden=decoder_hidden,
                          decoder_dropout=decoder_dropout, alphabet_size=len(alphabet),
                          n_conditions=n_conditions, activation=activation)
      super().__init__(n_conditions=n_conditions, autoregressive=False)

class ARVAE(BaseProtVAE):
  def __init__(self, upsample=False, original_dim=504, latent_dim=50,
               alphabet=aa_letters, batch_size=32, input_dropout=None,
               ncell=512, clipnorm=5, min_deconv_dim=42, lr=0.001, project_x=True):
    self.latent_dim = latent_dim
    self.original_dim = original_dim
    self.E = cnn_encoder(21, 2, original_dim,
                         len(alphabet), latent_dim, BN=True,
                         activation='prelu')
    self.S = sampler(latent_dim, epsilon_std=1.)
    self.G = recurrent_sequence_decoder(latent_dim, original_dim, ncell=ncell, upsample=upsample,
                                        input_dropout=input_dropout, min_deconv_dim=min_deconv_dim,
                                        batch_size=batch_size, alphabet_size=len(alphabet),
                                        intermediate_dim=63, project_x=project_x)
    super().__init__(autoregressive=True, clipnorm=clipnorm, lr=lr)
