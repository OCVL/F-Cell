import itertools
import sys
from pathlib import Path
from tkinter import filedialog, Tk
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

root = Tk()
root.lift()
w = 1
h = 1
x = root.winfo_screenwidth() / 4
y = root.winfo_screenheight() / 4
root.geometry('%dx%d+%d+%d' % (w, h, x, y))  # This moving around is to make sure the dialogs appear in the middle of the screen.

pName = None

pName = filedialog.askdirectory(title="Select the folder containing RMS data.", initialdir=None, parent=root)

if not pName:
    sys.exit(1)

searchpath = Path(pName)

rmsdata = []
rms_cov = []

for path in searchpath.glob("*.csv"):
    dataset = pd.read_csv(path, index_col=0, header=1)
    tmp = dataset.loc[:,"Amplitude"].to_numpy()
    print(path)
    rmsdata.append(tmp[:-1])
    rms_cov.append( np.nanstd(tmp[:-1])/np.nanmean(tmp[:-1]) )

rmsdata=np.array(list(itertools.zip_longest(*rmsdata, fillvalue=np.nan))).T
rms_cov = np.array(rms_cov)

pName = filedialog.askdirectory(title="Select the folder containing STDDEV data.", initialdir=None, parent=root)

if not pName:
    sys.exit(1)

searchpath = Path(pName)

stddata = []
std_cov = []

for path in searchpath.glob("*.csv"):
    dataset = pd.read_csv(path, index_col=0, header=1)

    tmp = dataset.loc[:, "Amplitude"].to_numpy()
    print(path)
    stddata.append(tmp[:-1])
    std_cov.append( np.nanstd(tmp[:-1])/np.nanmean(tmp[:-1]) )

stddata=np.array(list(itertools.zip_longest(*stddata, fillvalue=np.nan))).T
std_cov = np.array(std_cov)

normmap = mpl.colors.Normalize(vmin=0, vmax=rmsdata.shape[0], clip=True)
mapper = plt.cm.ScalarMappable(cmap=plt.get_cmap("inferno"), norm=normmap)

plt.figure("Amplitude comparison")
for r in range(rmsdata.shape[0]):
    plt.scatter(rmsdata[r,:], stddata[r,:], s=35.0 ,color=mapper.to_rgba(r), linewidth=0)

plt.ylabel("Cooper 2017 (std dev)")
plt.ylim((0,10))
plt.xlabel("Gaffney 2024 (rms)")
plt.xlim((0,50))

plt.figure("CoV comparison")
for r in range(rmsdata.shape[0]):
    plt.scatter(rms_cov[r], std_cov[r], color=mapper.to_rgba(r))
plt.plot([0, 0.5], [0, 0.5], c='k' )
plt.xlim((0,0.5))
plt.ylim((0,0.5))


plt.show()