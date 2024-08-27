##################################
## Auxiliary Functions
##################################
def binarize_data(X, th=127):
    # th: threshold
    X[X < th] = 0
    X[X >= th] = 1
    return X 