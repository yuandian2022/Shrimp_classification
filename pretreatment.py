#%%
import pandas as pd
import numpy as np
import torch
from inception_raman import *
from scipy.signal import savgol_filter
from scipy.signal import hilbert

draw_df = pd.read_csv('dataset_19504.csv')

# %% SG
sg_df = savgol_filter(draw_df.iloc[:, 5:], 11, 3)
sg_df = pd.DataFrame(sg_df, index=draw_df.index, columns=draw_df.columns[5:])
sg_df = pd.concat([draw_df.iloc[:, :5], sg_df], axis=1)

# %% SNV
def snv(spectrum_data):
    Xnir = spectrum_data
    Xm = np.mean(Xnir, axis=1, keepdims=True)
    dX = Xnir - Xm
    Xsnv = dX / np.sqrt(np.sum(dX**2, axis=1, keepdims=True) / Xnir.shape[1])
    return Xsnv

snv_df = snv(draw_df.iloc[:, 5:].values)
snv_df = pd.DataFrame(snv_df, columns=draw_df.columns[5:], index=draw_df.index)
snv_df = pd.concat([draw_df.iloc[:, :5], snv_df], axis=1)

# %% MSC
def msc(spectrum_data):
    n = spectrum_data.shape[0] 
    k = np.zeros(spectrum_data.shape[0])
    b = np.zeros(spectrum_data.shape[0])
    M = np.mean(spectrum_data, axis=0)
 
    from sklearn.linear_model import LinearRegression
    for i in range(n):
        y = spectrum_data[i, :]
        y = y.reshape(-1, 1)
        M = M.reshape(-1, 1)
        model = LinearRegression()
        model.fit(M, y)
        k[i] = model.coef_
        b[i] = model.intercept_
 
    spec_msc = np.zeros_like(spectrum_data)
    for i in range(n):
        bb = np.repeat(b[i], spectrum_data.shape[1])
        kk = np.repeat(k[i], spectrum_data.shape[1])
        temp = (spectrum_data[i, :] - bb)/kk
        spec_msc[i, :] = temp
    return spec_msc

msc_df = msc(draw_df.iloc[:, 5:].values)
msc_df = pd.DataFrame(msc_df, columns=draw_df.columns[5:], index=draw_df.index)
msc_df = pd.concat([draw_df.iloc[:, :5], msc_df], axis=1)

# %% HT
spectrum_data = draw_df.iloc[:, 5:]
hilbert_transformed_data = np.abs(hilbert(spectrum_data, axis=1))
hilbert_df = pd.DataFrame(hilbert_transformed_data, columns=spectrum_data.columns, index=spectrum_data.index)
hilbert_df = pd.concat([draw_df.iloc[:, :5], hilbert_df], axis=1)

# %% subtract_baseline
def subtract_baseline(spectrum_data, poly_order=3):
    n, p = spectrum_data.shape
    baseline = np.zeros((n, p))
    for i in range(n):
        x = np.arange(p)
        y = spectrum_data[i]
        z = np.polyfit(x, y, poly_order)
        poly = np.poly1d(z)
        baseline[i] = poly(x)
    return spectrum_data - baseline

sub_df = subtract_baseline(draw_df.iloc[:, 5:].values)
sub_df = pd.DataFrame(sub_df, columns=draw_df.columns[5:], index=draw_df.index)
sub_df = pd.concat([draw_df.iloc[:, :5], sub_df], axis=1)