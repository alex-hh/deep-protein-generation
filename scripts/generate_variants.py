import argparse
import pickle

import numpy as np

from models.vaes import MSAVAE, ARVAE
from utils.io import output_fasta


def main(weights_file, msa=True, num_samples=500, output_file=None, model_kwargs=None,
         temperature=0., posterior_var_scale=1., solubility_level=None):

  if output_file is None:
    base_name = weights_file.split('/')[-1].split('.')[0]
    output_file = 'output/generated_sequences/{}_variants.fa'.format(base_name)

  if model_kwargs is None:
    model_kwargs = {}
  else:
    with open(model_kwargs, 'rb') as p:
      model_kwargs = pickle.load(p)

  if msa:
    model = MSAVAE(**model_kwargs)
  else:
    model = ARVAE(**model_kwargs)

  model.load_weights(weights_file)

  variants = model.generate_variants_luxA(num_samples, posterior_var_scale=posterior_var_scale,
                                          temperature=temperature, solubility_level=solubility_level)
  
  names = ['luxa_var{}'.format(i+1) for i in range(num_samples)]
  output_fasta(names, variants, output_file)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('weights_file', type=str)
  parser.add_argument('--unaligned', action='store_true')
  parser.add_argument('--output_file', default=None, type=str)
  parser.add_argument('--num_samples', default=500, type=int)
  parser.add_argument('--model_kwargs', default=None, type=str)
  parser.add_argument('--solubility_level', default=None, type=int)
  parser.add_argument('--temperature', default=0., type=float)
  parser.add_argument('--var_scale', default=1., type=float)
  args = parser.parse_args()
  main(args.weights_file, msa=not args.unaligned, output_file=args.output_file,
       model_kwargs=args.model_kwargs, temperature=args.temperature,
       posterior_var_scale=args.var_scale, solubility_level=args.solubility_level)