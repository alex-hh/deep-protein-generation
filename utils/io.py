import gzip


def load_gzdata(filename, one_hot=True):
    names, seqs = read_gzfasta(filename)
    if one_hot:
        seqs = to_one_hot(seqs)
    return names, seqs

def compress_file(filename):
    with open(filename, 'rb') as f:
        with open(filename+'.gz', 'wb') as gzout:
            b = gzip.compress(f.read())
            gzout.write(b)

def read_gzfasta(filepath, output_arr=False, encoding='utf-8'):
    names = []
    seqs = []
    seq = ''
    with gzip.open(filepath, 'rb') as fin:
        file_content = fin.read().decode(encoding)
        lines = file_content.split('\n')
        for line in lines:
            if line:
                if line[0] == '>':
                    if seq:
                        names.append(name)
                        if output_arr:
                            seqs.append(np.array(list(seq)))
                        else:
                            seqs.append(seq)
                    name = line[1:]
                    seq = ''
                else:
                    seq += line
            else:
                continue
        if seq: # handle last seq in file
            names.append(name)
            if output_arr:
                seqs.append(np.array(list(seq)))
            else:
                seqs.append(seq)
    if output_arr:
        seqs = np.array(seqs)
    return names, seqs

def read_fasta(filepath, output_arr=False):
    names = []
    seqs = []
    seq = ''
    with open(filepath, 'r') as fin:
        for line in fin:
            if line[0] == '>':
                if seq:
                    names.append(name)
                    if output_arr:
                        seqs.append(np.array(list(seq)))
                    else:
                        seqs.append(seq)
                name = line.rstrip('\n')[1:]
                seq = ''
            else:
                seq += line.rstrip('\n')
        if seq: # handle last seq in file
            names.append(name)
            if output_arr:
                seqs.append(np.array(list(seq)))
            else:
                seqs.append(seq)
    if output_arr:
        seqs = np.array(seqs)
    return names, seqs

def output_fasta(names, seqs, filepath):
    with open(filepath, 'w') as fout:
        for name, seq in zip(names, seqs):
            fout.write('>{}\n'.format(name))
            fout.write(seq+'\n')