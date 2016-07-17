from astropy.modeling.fitting import SherpaFitter
<<<<<<< HEAD
sfitter = SherpaFitter(statistic='chi2', optimizer='levmar', estmethod='covariance')
=======
sfitter = SherpaFitter(statistic='chi2', optimizer='levmar', estmethod='convariance')
>>>>>>> b70820ceda946bb9d61bda09ee745308050a94ba


from astropy.modeling.models import Gaussian1D
import numpy as np
import  matplotlib.pyplot as plt

np.random.seed(0x1337)

true = Gaussian1D(amplitude=3, mean=0.9, stddev=0.5)
err = 0.8
step = 0.2
x = np.arange(-3, 3, step)
y = true(x) + err * np.random.uniform(-1, 1, size=len(x))

yerrs=err * np.random.uniform(0.2, 1, size=len(x))
<<<<<<< HEAD
#binsize=step * np.ones(x.shape)  # please note these are binsize/2 not true errors! 
binsize=(step/2) * np.ones(x.shape)  # please note these are binsize/2 not true errors! 
=======
binsize=step * np.ones(x.shape)  # please note these are binsize/2 not true errors! 
>>>>>>> b70820ceda946bb9d61bda09ee745308050a94ba

fit_model = true.copy() # ofset fit model from true 
fit_model.amplitude = 2
fit_model.mean = 0
fit_model.stddev = 0.2

plt.plot(x,true(x), label="True")
<<<<<<< HEAD
plt.errorbar(x, y, xerr=binsize, yerr=yerrs, ls="", label="Data")
=======
plt.errorbar(x, y, binsize=binsize, yerr=yerrs, ls="", label="Data")
>>>>>>> b70820ceda946bb9d61bda09ee745308050a94ba
plt.plot(x,fit_model(x), label="Starting fit model")
plt.legend(loc=(0.02,0.7), frameon=False)
plt.xlim((-3,3))
plt.savefig("_generated/example_plot_data.png")
plt.close('all')
<<<<<<< HEAD


fitted_model = sfitter(fit_model,x, y, xbinsize=binsize, err=yerrs)

plt.plot(x,true(x), label="True")
plt.errorbar(x, y, xerr=binsize, yerr=yerrs, ls="", label="Data")
plt.plot(x,fit_model(x), label="Starting fit model")
plt.plot(x,fitted_model(x), label="Fitted model")
plt.legend(loc=(0.02,0.6), frameon=False)
plt.xlim((-3,3));
plt.savefig("_generated/example_plot_fitted.png")
plt.close('all')


param_errors = sfitter.est_errors(sigma=3)
min_model = fitted_model.copy()
max_model = fitted_model.copy()

for pname, pval, pmin, pmax in zip([p.split(".",1)[-1] for p in param_errors.parnames], param_errors.parvals, 
                                   param_errors.parmins, param_errors.parmaxes):
    getattr(min_model,pname).value = pval+pmin
    getattr(max_model,pname).value = pval+pmax


plt.plot(x,true(x), label="True")
plt.errorbar(x, y, xerr=binsize, yerr=yerrs, ls="")
plt.plot(x,fitted_model(x), label="Fitted model")
plt.plot(x,min_model(x), label="min model", ls="--")
plt.plot(x,max_model(x), label="max model", ls="--")
plt.legend(loc=(0.02,0.6), frameon=False)
_ = plt.xlim((-3,3))

plt.savefig("_generated/example_plot_error.png")
plt.close('all')



double_gaussian = Gaussian1D(amplitude=7, mean=-1.5, stddev=0.5) + Gaussian1D(amplitude=1, mean=0.9, stddev=0.5)

def tiedfunc(self): # a function used for tying amplitude_1
    return 1.2*self.amplitude_0

double_gaussian.amplitude_1.tied = tiedfunc
double_gaussian.amplitude_1.value=double_gaussian.amplitude_1.tied(double_gaussian)


err = 0.8
step = 0.2
x = np.arange(-3, 3, step)
y = double_gaussian(x) + err * np.random.uniform(-1, 1, size=len(x))
yerrs = err * np.random.uniform(0.2, 1, size=len(x))
binsize=(step/2) * np.ones(x.shape)  # please note these are binsize/2 not true errors! 



plt.errorbar(x, y, xerr=binsize, yerr=yerrs,ls="", label="data") 
#once again xerrs are binsize/2 not true errors! 
plt.plot(x,double_gaussian(x),label="True")
plt.legend(loc=(0.78,0.8), frameon=False)
_ = plt.xlim((-3,3))


plt.savefig("_generated/example_plot_data2.png")
plt.close('all')



fit_gg = double_gaussian.copy()
fit_gg.mean_0.value = -0.5
fit_gg.mean_0.min = -1.25 # sets the lower bound so we can force the parameter against it
fit_gg.mean_1.value = 0.8
fit_gg.stddev_0.value = 0.9
fit_gg.stddev_0.fixed = True

fitted_gg = sfitter(fit_gg,x, y, xbinsize=binsize, err=yerrs)
print("##Fit with contraints")
print(sfitter._fitmodel.sherpa_model)

free_gg = sfitter(double_gaussian.copy(),x, y, xbinsize=binsize, err=yerrs)
print
print("##Fit without contraints")
print(sfitter._fitmodel.sherpa_model)

plt.figure(figsize=(10,5))
plt.plot(x,double_gaussian(x),label="True")
plt.errorbar(x, y, xerr=binsize, yerr=yerrs,ls="", label="data")
plt.plot(x,fit_gg(x),label="Pre fit")
plt.plot(x,fitted_gg(x),label="Fitted")
plt.plot(x,free_gg(x),label="Free")
plt.subplots_adjust(right=0.8)
plt.legend(loc=(1.01,0.55), frameon=False)
plt.xlim((-3,3))

plt.savefig("_generated/example_plot_fitted2.png")
plt.close('all')


fit_gg = double_gaussian.copy()
fit_gg.mean_0.value = -0.5
fit_gg.mean_0.min = -1.25
fit_gg.mean_1.value = 0.8
fit_gg.stddev_0.value = 0.9
fit_gg.stddev_0.fixed = True

fm1,fm2 = sfitter([fit_gg, double_gaussian.copy()], x, y, xbinsize=binsize, err=yerrs)


plt.figure(figsize=(10,5))
plt.plot(x,double_gaussian(x),label="True")
plt.errorbar(x, y, xerr=binsize, yerr=yerrs,ls="", label="data")
plt.plot(x, fit_gg(x),label="Pre fit")
plt.plot(x, fm1(x),label="Constrained")
plt.plot(x, fm2(x),label="Free")
plt.subplots_adjust(right=0.8)


plt.legend(loc=(1.01,0.55), frameon=False)
plt.xlim((-3,3))


plt.savefig("_generated/example_plot_simul.png")
plt.close("all")



fit_gg=double_gaussian.copy()
fit_gg.mean_0 = -2.3
fit_gg.mean_1 = 0.7
fit_gg.amplitude_0 = 2
fit_gg.amplitude_1 = 3
fit_gg.stddev_0 = 0.3
fit_gg.stddev_1 = 0.5


second_gg = double_gaussian.copy()
second_gg.mean_0 = -2
second_gg.mean_1 = 0.5
second_gg.amplitude_0 = 8
second_gg.amplitude_1 = 5
second_gg.stddev_0 = 0.4
second_gg.stddev_1 = 0.8
second_gg.amplitude_1.value=second_gg.amplitude_1.tied(second_gg)


yy2 = second_gg(x) + err * np.random.uniform(-1, 1, size=len(x))
yy2errs = err * np.random.uniform(0.2, 1, size=len(x))

plt.errorbar(x, y, xerr=binsize, yerr=yerrs,ls="", label="data1")
plt.errorbar(x, yy2, yerr=yy2errs,ls="", label="data2")
plt.plot(x, fit_gg(x), label="Prefit")

fitted_model = sfitter(fit_gg, x=[x, x], y=[y, yy2], xbinsize=[binsize, None], err=[yerrs, yy2errs])

plt.plot(x, fitted_model[0](x), label="Fitted")
plt.plot(x, fitted_model[1](x), label="Fitted")
plt.subplots_adjust(right=0.8)


plt.legend(loc=(1.01,0.55), frameon=False)
plt.xlim((-3,3))


plt.savefig("_generated/example_plot_simul2.png")
plt.close("all")


















print("Done")
=======
>>>>>>> b70820ceda946bb9d61bda09ee745308050a94ba
