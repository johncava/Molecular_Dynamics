import numpy as np

def getBucket(chunk, bucketSize=1000):
    # Chunk 1: 0 to 1000
    # Chunk 2: 980 to 2000
    # Chunk 3: 1980 to 3000
    # etc.

    if (chunk-1):
        goback = 20
    else:
        goback = 0
    trunc_start = (chunk - 1) * bucketSize - goback
    trunc_stop = chunk * bucketSize
    return int(trunc_start), int(trunc_stop)

