import numpy as np

def getChunk(wround):
    if wround < 98:
        chunk = 1
    else:
        temp = wround + 2
        temp2 = temp % 100
        temp3 = (temp - temp2)/100
        chunk = temp3 + 1
    return int(chunk)

