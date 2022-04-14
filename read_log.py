from pathlib import Path
from collections import OrderedDict

files = list(Path('log_base').glob('*'))

k = OrderedDict()

for i in files:
    with i.open() as f:
        z = f.readlines()[-1]
    acc = float(z.split('\t')[-1].split('\n')[0])
    name = i.stem
    k[name] = acc

for i in sorted(k.keys(),key=int) : 
    print('{} : {}'.format(i,k[i]))

# print(k.items())