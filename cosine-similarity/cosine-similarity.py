import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    a = np.asarray(a, dtype = float)
    b = np.asarray(b, dtype = float)

    num = np.dot(a,b)
    a_mod = np.sqrt(np.dot(a,a))
    b_mod = np.sqrt(np.dot(b,b))
    if(a_mod == 0 or b_mod == 0):
        return 0.0
    return float(num/(a_mod*b_mod))
    