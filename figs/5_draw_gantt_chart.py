import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


'''=========================================================Main drawing code=========================================================='''


if __name__ == '__main__':
    import pickle
    with open('figs/gantt/gantt_data.pkl', 'rb') as f:
        dic = pickle.load(f)
    print(dic)

    
    
    

