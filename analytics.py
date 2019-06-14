import numpy as np



def analytics(array,length):
    print(array)
    max_array=max(array)
    max_array=int(max_array)
    if(max_array>=20):
        new_array=np.zeros(max_array+1)
    else:
        new_array=np.zeros(20)
    print(len(new_array))
    for i in range(0, length):
        new_array[int(array[i])]= new_array[int(array[i])] + 1
    return new_array