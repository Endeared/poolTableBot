import os

toClean = './annotateImgs2/'
test = os.listdir(toClean)

for item in test:
    if item.endswith(".txt"):
        os.remove(os.path.join(toClean, item))