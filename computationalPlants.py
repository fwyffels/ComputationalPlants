import numpy as np
import pylab as pl
import seaborn as sb
from sklearn import linear_model

STEPS_FOR_TESTING = 24*60*1.5
A_DAY = 60*24

# data preparation
data = np.genfromtxt('20140907_data_plants_trial.csv', delimiter=',')
Y = data[1:,1]
Y[np.isnan(Y)]=0 # replace nan's by zero, better use interpolation!!!
X = data[1:,2:]
X_norm=(X-np.mean(X,0))/np.std(X,0)
Y_norm=(Y-np.mean(Y))/np.std(Y)

# EXPERIMENT: memory versus prediction RIDGE REGRESSION
# train and test data
Y_norm_train = Y_norm[A_DAY:-A_DAY-STEPS_FOR_TESTING-1]
Y_norm_test = Y_norm[-A_DAY-STEPS_FOR_TESTING:-A_DAY-1]

clf = linear_model.Ridge(fit_intercept=False,normalize=False,alpha=0.1)
MAE_train = []
MAE_test = []
predictions = []
for time in range(-A_DAY,A_DAY):
    X_norm_train = X_norm[A_DAY-time:-A_DAY-STEPS_FOR_TESTING-1-time,:]   # -time: we go from fitting the past to prediction
    X_norm_test = X_norm[-A_DAY-STEPS_FOR_TESTING-time:-A_DAY-1-time,:]
    clf.fit(X_norm_train, Y_norm_train)
    # train error
    Y_pred = clf.predict(X_norm_train)
    MAE_train.append(np.mean(np.absolute(Y_pred-Y_norm_train)))
    # test error
    Y_pred = clf.predict(X_norm_test)
    predictions.append(Y_pred)
    MAE_test.append(np.mean(np.absolute(Y_pred-Y_norm_test)))

# plot MAE in function of time
ax = pl.gca()
ax.set_color_cycle(['b', 'r'])

ax.plot(range(-A_DAY,A_DAY), MAE_train)
ax.plot(range(-A_DAY,A_DAY), MAE_test)
pl.xlabel('Time shift in minutes')
pl.ylabel('MAE')
pl.title('MAE train (blue) and test (red) as a function of the time shift')
pl.axis('tight')
pl.show()

# EXPERIMENT: memory versus prediction with LASSO
# train and test data
Y_norm_train = Y_norm[A_DAY:-A_DAY-STEPS_FOR_TESTING-1]
Y_norm_test = Y_norm[-A_DAY-STEPS_FOR_TESTING:-A_DAY-1]

clf = linear_model.Lasso(fit_intercept=False,normalize=False,alpha=0.01)
MAE_train = []
MAE_test = []
predictions = []
coefs = []
for time in range(-A_DAY,A_DAY):
    X_norm_train = X_norm[A_DAY-time:-A_DAY-STEPS_FOR_TESTING-1-time,:]   # -time: we go from fitting the past to prediction
    X_norm_test = X_norm[-A_DAY-STEPS_FOR_TESTING-time:-A_DAY-1-time,:]
    clf.fit(X_norm_train, Y_norm_train)
    coefs.append(clf.coef_)
    # train error
    Y_pred = clf.predict(X_norm_train)
    MAE_train.append(np.mean(np.absolute(Y_pred-Y_norm_train)))
    # test error
    Y_pred = clf.predict(X_norm_test)
    predictions.append(Y_pred)
    MAE_test.append(np.mean(np.absolute(Y_pred-Y_norm_test)))

# plot MAE in function of time
ax = pl.gca()
ax.set_color_cycle(['b', 'r'])

ax.plot(range(-A_DAY,A_DAY), MAE_train)
ax.plot(range(-A_DAY,A_DAY), MAE_test)
pl.xlabel('Time shift in minutes')
pl.ylabel('MAE')
pl.title('MAE train (blue) and test (red) as a function of the time shift')
pl.axis('tight')
pl.show()

# plot coefficients in function of time
ax = pl.gca()
ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])

ax.plot(range(-A_DAY,A_DAY), coefs)
ax.plot(range(-A_DAY,A_DAY), coefs)
pl.xlabel('Time shift in minutes')
pl.ylabel('coefficients')
pl.title('coefficients in function of shifted time')
pl.axis('tight')
pl.show()

# EXPERIMENT: memory versus prediction with LARS
# train and test data
Y_norm_train = Y_norm[A_DAY:-A_DAY-STEPS_FOR_TESTING-1]
Y_norm_test = Y_norm[-A_DAY-STEPS_FOR_TESTING:-A_DAY-1]

MAE_train = []
MAE_test = []
predictions = []
coefs = []
for time in range(-A_DAY,A_DAY,60):
    print time
    X_norm_train = X_norm[A_DAY-time:-A_DAY-STEPS_FOR_TESTING-1-time,:]   # -time: we go from fitting the past to prediction
    alphas, _, coefs = linear_model.lars_path(X_norm_train, Y_norm_train, method='lasso', verbose=True)

    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]

    ax = pl.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
    pl.plot(xx, coefs.T)
    ymin, ymax = pl.ylim()
    pl.vlines(xx, ymin, ymax, linestyle='dashed')
    pl.xlabel('|coef| / max|coef|')
    pl.ylabel('Coefficients')
    pl.title('LASSO Path')
    pl.axis('tight')
    pl.show()

# EXPERIMENT: ridge regression with optimization of regul parameter
# train and test data
X_norm_train = X_norm[0:-STEPS_FOR_TESTING,:]
Y_norm_train = Y_norm[0:-STEPS_FOR_TESTING]

X_norm_test = X_norm[-STEPS_FOR_TESTING:,:]
Y_norm_test = Y_norm[-STEPS_FOR_TESTING:]

regul_a = 100
regul_a = np.logspace(-5, 0, regul_a)
clf = linear_model.Ridge(fit_intercept=False,normalize=False)
coefs = []
predictions = []
MAE_train = []
MAE_test = []
for a in regul_a:
    clf.set_params(alpha=a)
    clf.fit(X_norm_train, Y_norm_train)
    coefs.append(clf.coef_)
    # train error
    Y_pred = clf.predict(X_norm_train)
    MAE_train.append(np.mean(np.absolute(Y_pred-Y_norm_train)))
    # test error
    Y_pred = clf.predict(X_norm_test)
    predictions.append(Y_pred)
    MAE_test.append(np.mean(np.absolute(Y_pred-Y_norm_test)))

# plot MAE in function of regul parameter
ax = pl.gca()
ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])

ax.plot(regul_a, MAE_train)
ax.plot(regul_a, MAE_test)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
pl.xlabel('alpha')
pl.ylabel('MAE')
pl.title('MAE train (blue) and test (red) as a function of the regularization')
pl.axis('tight')
pl.show()

# plot predictions
ax = pl.gca()
ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])
pl.plot(Y_norm_test)
pl.plot(predictions[0])
pl.plot(predictions[40])
pl.plot(predictions[80])
pl.plot(predictions[120])
pl.plot(predictions[160])
pl.plot(predictions[199])
pl.show()