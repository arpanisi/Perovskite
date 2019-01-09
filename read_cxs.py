import os, sys
from cxsParser import HirshfeldSurface as hs
import matplotlib.pyplot as plt
import matplotlib.colors as cols
import numpy as np
import cv2


ROOT_DIR = os.getcwd()
PEROV_DIR = os.path.join(ROOT_DIR, 'perovskite')
NON_PEROV_DIR = os.path.join(ROOT_DIR, 'non-perovskite')
# filelist = []
imgs = []

bins = 32

for f in os.listdir(PEROV_DIR):
     if f.endswith('cxs'):
         filename = os.path.join(PEROV_DIR, f)
         hsurface = hs(filename)
         img, _, _ = np.histogram2d(hsurface.d_e, hsurface.d_i, bins=bins)
         imgs.append(img)
         # filelist.append(f)

for f in os.listdir(NON_PEROV_DIR):
     if f.endswith('cxs'):
         filename = os.path.join(NON_PEROV_DIR, f)
         hsurface = hs(filename)
         img, _, _ = np.histogram2d(hsurface.d_e, hsurface.d_i, bins=bins)
         imgs.append(img)

labels = np.r_[np.zeros(len(os.listdir(PEROV_DIR))), np.ones(len(os.listdir(NON_PEROV_DIR)))]
imgs = np.asarray(imgs)
np.save('fingerprint.npy', imgs)
np.save('labels.npy', labels)

# f = os.path.join(CXS_DIR, 'La3Co4Sn13.cxs')
# hsurface = hs(f)

# img, _, _ = np.histogram2d(hsurface.d_e, hsurface.d_i, bins=251)
# plt.hist2d(hsurface.d_i, hsurface.d_e, norm=cols.LogNorm(), bins=32)

# canvas = hsurface.hirshfeld_surf()
