import sys
sys.path.append('DOTA_devkit')
from ImgSplit import splitbase

# We start by splitting the train set
split = splitbase(r'dota/train', r'splitted_dota/train', choosebestpoint=True)
split.splitdata(2)

# We then split the test set
split = splitbase(r'dota/val', r'splitted_dota/val', choosebestpoint=True)
split.splitdata(2)


