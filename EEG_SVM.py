#import the mne,scikit and numpy modules to read the EEG data
#All data have been pre-procesed with EEGLAB in MATLAB, therefore  the format of data is .set

import mne
import scipy
import matplotlib.pyplot as plt
from mne.datasets import testing
from mne import Epochs, io, pick_types
from mne.event import define_target_events
import numpy as np
from mne.minimum_norm import read_inverse_operator, compute_source_psd
from mne.time_frequency import psd_multitaper
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score

#function to draw the comparison of some kernel SVM
def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy



def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
#location of dataset,change it to related with your localhost
data_path = testing.data_path()

low_beta_count = [[0 for x in range(2)] for y in range(40)]
middle_beta_count = [[0 for x in range(2)] for y in range(40)]
high_beta_count = [[0 for x in range(2)] for y in range(40)]
index=[0 for x in range(40)]
 
#our recording standard: 10-20
montage = mne.channels.read_montage('standard_1020')
k=0
#i is the subject id
for i in range(1,6):
    
    for j in range(1,5):
        #read for word-task, j is the trial code
        fname = data_path + "/Dataset/d" + str(i) + "-ep1-"+ str(j) + ".set"
        raw = io.eeglab.read_raw_eeglab(fname, montage=montage)

        #measure the distribution of low-beta
        psds, freqs = psd_multitaper(raw, low_bias=True, 
                             fmin=12, fmax=16, proj=True, 
                             n_jobs=1)
        psds = 10 * np.log10(psds)
        low_beta_count[k][0] = psds.mean()
        low_beta_count[k][1] = psds.std()
        
        #measure the distribution of middle-beta
        psds, freqs = psd_multitaper(raw, low_bias=True, 
                             fmin=16, fmax=22, proj=True,
                             n_jobs=1)
        psds = 10 * np.log10(psds)
        middle_beta_count[k][0] = psds.mean()
        middle_beta_count[k][1] = psds.std()
        
        #measure the distribution of high-beta
        psds, freqs = psd_multitaper(raw, low_bias=True, 
                             fmin=22, fmax=30, proj=True, 
                             n_jobs=1)
        psds = 10 * np.log10(psds)
        high_beta_count[k][0] = psds.mean()
        high_beta_count[k][1] = psds.std()
        
        #index 1 for word task
        index[k]=1
        k=k+1
        
        #read for word-task, j is the trial code
        fname = data_path + "/EEGLAB/d" + str(i) + "-ep2-"+ str(j) + ".set"
        raw = io.eeglab.read_raw_eeglab(fname, montage=montage)

        
        
        
        #measure the distribution of low-beta
        psds, freqs = psd_multitaper(raw, low_bias=True,
                             fmin=12, fmax=16, proj=True,
                             n_jobs=1)
        psds = 10 * np.log10(psds)
        low_beta_count[k][0] = psds.mean()
        low_beta_count[k][1] = psds.std()
       
        
        #measure the distribution of middle-beta
        psds, freqs = psd_multitaper(raw, low_bias=True, 
                             fmin=16, fmax=22, proj=True,
                             n_jobs=1)
        psds = 10 * np.log10(psds)
        middle_beta_count[k][0] = psds.mean()
        middle_beta_count[k][1] = psds.std()
        
        
        
        
        #measure the distribution of high-beta
        psds, freqs = psd_multitaper(raw, low_bias=True,
                             fmin=22, fmax=30, proj=True, 
                             n_jobs=1)
        psds = 10 * np.log10(psds)
        high_beta_count[k][0] = psds.mean()
        high_beta_count[k][1] = psds.std()
        
        
        #index 2 for color task task
        index[k]=2
        k=k+1

#rejected 3 data of color-task
index[6]=0
index[14]=0
index[28]=0        


#convert to numpy format
low_beta_count=np.array(low_beta_count)
middle_beta_count=np.array(middle_beta_count)
high_beta_count=np.array(high_beta_count)
index=np.array(index)        

#we try 4 type of SVm machine in the low beta data
#ytested the middle beta -> change low to middle
#tested the high beta -> change low to high
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='poly', degree=3, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(low_beta_count, index) for clf in models)

fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
X0, X1 = low_beta_count[:,0], low_beta_count[:,1]

xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=index, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Mean')
    ax.set_ylabel('Std')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
#measure the accuracy, example in high beta
clf = svm.SVC(kernel='poly', degree=4, C=C)
clf.fit(high_beta_count, index)    
predictor=np.array([])
for i in range(40):
    predictor[i]=0
for i in range(40):
    tester=high_beta_count[i,:]
    predictor[i]=clf.predict([tester])
    
print(accuracy_score(predictor, index))
    
        
    
