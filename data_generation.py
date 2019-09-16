import os
import pandas as pd
from cxsParser import HirshfeldSurface as hs
import numpy as np


def generate_data(bins=32, value=None):
    imgs = np.load('new_perovskites.npy')
    list_of_perovskite = np.load('new_names.npy')

    imgs = [img for img in imgs]
    input = pd.DataFrame({'Chemical formula': list_of_perovskite, 'Imgs': imgs})

    list_of_structures = pd.read_csv('list_of_structures.csv')

    final = input.merge(list_of_structures, on='Chemical formula')

    cols = final.columns

    #values_of_interest = ['formation', 'band gap', 'stability']
    output_col_name = cols[cols.str.match(value, case=False)]

    ind = np.where(final[output_col_name] != '-')[0]
    final = final.iloc[ind]

    final = final[['Chemical formula', 'Imgs', output_col_name[0]]]

    return final
