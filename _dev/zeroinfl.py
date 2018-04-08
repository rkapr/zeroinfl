import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy as sp
import scipy.stats as st
import sys
import warnings
import inspect
import patsy
from decimal import Decimal

FLOAT_EPS = np.finfo(float).eps
pd.options.display.float_format = '{:,.12f}'.format

__all__ = ['ZeroInflated']


class LinkClass(object):
    def __init__(self):
        return NotImplementedError
    def link(self, mu):
        return NotImplementedError
    def link_inv(self, eta):
        return NotImplementedError
    def link_inv_deriv(self, eta):
        return NotImplementedError
        
    
class Logit(LinkClass):
    def __init__(self):
        self.linkclass = sm.genmod.families.links.logit
    def link(self, p):
        return np.log(p/(1.0-p))
        #return sp.special.logit(p)
    def link_inv(self, eta):
        thresh = 30.0
        eta = np.minimum(np.maximum(eta,-thresh), thresh)
        exp_eta = np.exp(-eta)
        return 1.0/(1.0+exp_eta)
        #return sp.special.expit(eta)
    def link_inv_deriv(self, eta):
        thresh = 30.0
        eta[abs(eta) > thresh] = FLOAT_EPS
        return np.exp(eta)/(1+np.exp(eta))**2
    def __repr__(self):
        display_string = f"\n    linkstr: logit"
        display_string += '\n    link: log(p/(1-p))'
        display_string += '\n    linkinv: exp(eta)/(1+exp(eta))'
        return display_string

class Probit(LinkClass):
    def __init__(self):
        self.linkclass = sm.genmod.families.links.probit
    def link(self, mu):
        return st.norm.ppf(mu)
    def link_inv(self, eta):
        thresh = -st.norm.ppf(FLOAT_EPS)
        eta = np.minimum(np.maximum(eta,-thresh),thresh)
        return st.norm.cdf(eta)
    def link_inv_deriv(self, eta):
        return np.maximum(st.norm.pdf(eta),FLOAT_EPS)
    def __repr__(self):
        display_string = f"\n    linkstr: probit"
        display_string += '\n    link: norm.ppf(mu)'
        display_string += '\n    linkinv: norm.cdf(eta)'
        return display_string
    
class CLogLog(LinkClass):
    def __init__(self):
        self.linkclass = sm.genmod.families.links.cloglog
    def link(self, mu):
        return np.log(-np.log(1 - mu))
    def link_inv(self, eta):
        return np.maximum(np.minimum(-np.expm1(-np.exp(eta)),1-FLOAT_EPS),FLOAT_EPS)
    def link_inv_deriv(self, eta):
        eta = np.minimum(eta,700)
        return np.maximum(np.exp(eta)*np.exp(-np.exp(eta)),FLOAT_EPS)
    def __repr__(self):
        display_string = f"\n    linkstr: cloglog"
        display_string += '\n    link: log(-log(1 - mu))'
        display_string += '\n    linkinv: 1-exp(-exp(eta))'
        return display_string
    
class Cauchit(LinkClass):
    def __init__(self):
        self.linkclass = sm.genmod.families.links.cauchy
    def link(self, mu):
        return st.cauchy.ppf(mu)
    def link_inv(self, eta):
        thresh = -st.cauchy.ppf(FLOAT_EPS)
        eta = np.minimum(np.maximum(eta,-thresh),thresh)
        return st.cauchy.cdf(eta)
    def link_inv_deriv(self, eta):
        return nnp.maximum(st.cauchy.pdf(eta),FLOAT_EPS)
    def __repr__(self):
        display_string = f"\n    linkstr: cauchit"
        display_string += '\n    link: cauchy.ppf(mu)'
        display_string += '\n    linkinv: cauchy.cdf(eta)'
        return display_string
    
class Log(LinkClass):
    def __init__(self):
        self.linkclass = sm.genmod.families.links.log
    def link(self, mu):
        return np.log(mu)
    def link_inv(self, eta):
        return np.maximum(np.exp(eta), FLOAT_EPS)
    def link_inv_deriv(self, eta):
        return np.maximum(np.exp(eta), FLOAT_EPS)
    def __repr__(self):
        display_string = f"\n    linkstr: log"
        display_string += '\n    link: log(mu)'
        display_string += '\n    linkinv: exp(eta)'
        return display_string
    


class ZeroInflated(object):
    __doc__ = """
    Zero Inflated model for count data
    %(params)s
    %(extra_params)s
    Attributes
    -----------
    formula_str : string
        A reference to the endogenous response variable.
    data : pandas dataframe
        A reference to the exogenous design.
    dist: string
        A reference to the zero-inflated exogenous design.
    link: string
        A reference to 
    """
    def __init__(self, formula_str, data, dist = 'poisson', offsetx = None, offsetz = None,
                 link = 'logit', weights = None, missing='none', **kwargs):
        self._set_data(formula_str, data, missing)
        self.terms = {'Y':self.endog.columns.values[0],'X':self.X.columns.values,\
                      'Z':self.Z.columns.values}
        self.formula = formula_str
        self.dist = self._dist_processing(dist)
        self.link = self._link_processing(link)
        self.n = len(self.endog)
        self._set_wt_offset(weights, offsetx, offsetz)
        self.linkobj = self._LinkClass_processing(self.link)
        self._set_loglik(self.dist)
        self.call = f"ZeroInflated(formula_str='{formula_str}', data={self._retrieve_name(data)}, dist='{dist}', offsetx={offsetx}, offsetz={offsetz},"
        self.call = self.call + f" link='{link}', weights={weights}, missing='{missing}')"
        
        # Convenience variables
        self.kx = self.X.shape[1]
        self.kz = self.Z.shape[1]
        self.Y = np.squeeze(self.endog)
        self.Y0 = self.Y <= 0
        self.Y1 = self.Y > 0
        self.EM = True
        self.reltol = (np.finfo(float).eps)**(1/1.6)       
        
        
    def print_obj(self):
        print(self)   
        



    def _retrieve_name(self, var):
        """
        Gets the name of var. Does it from the out most frame inner-wards.
        :param var: variable to get name from.
        :return: string
        """
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0:
                return names[-1]
        
    def _set_data(self, formula_str, data, missing):
        self.endog, self.X, self.Z = self.formula_processing(formula_str, data, missing=missing) 
     
           
    def _set_wt_offset(self, weights, offsetx, offsetz):
        ## weights and offset
        
        if weights is None:
            weights = 1.0
        weights = np.ndarray.flatten(np.array(weights))
        if weights.size == 1:
            weights = np.repeat(weights,self.n)
        weights = pd.Series(data = weights, index = self.X.index)

        if offsetx is None:
            offsetx = 0.0
        offsetx = np.ndarray.flatten(np.array(offsetx))
        if offsetx.size == 1:
            offsetx = np.repeat(offsetx,self.n)

        if offsetz is None:
            offsetz = 0.0
        offsetz = np.ndarray.flatten(np.array(offsetz))
        if offsetz.size == 1:
            offsetz = np.repeat(offsetz,self.n)
        
        self.weights = weights
        self.offsetx = offsetx
        self.offsetz = offsetz
        
    def _set_loglik(self, dist):
        if dist is 'poisson':
            self.loglikfun = self.ziPoisson
            self.gradfun = self.gradPoisson
        elif dist is 'negbin':
            self.loglikfun = self.ziNegBin
            self.gradfun = self.gradNegBin
        else:
            self.loglikfun = self.ziGeom
            self.gradfun = self.gradGeom
        
    def ziPoisson(self, parms, sign = 1.0):
        """
        Log-likelihood of Zero Inflated Poisson.
        """
        
        ## count mean
        mu = np.exp(np.dot(self.X,parms[np.arange(self.kx)]) + self.offsetx)
        ## binary mean
        phi = self.linkobj.link_inv(np.dot(self.Z, parms[np.arange(self.kx,self.kx+self.kz)]) +\
                                    self.offsetz)
        ## log-likelihood for y = 0 and y >= 1
        loglik0 = np.log(phi + np.exp(np.log(1-phi) - mu)) 
        loglik1 = np.log(1-phi) + sp.stats.poisson.logpmf(self.Y, mu)
        ## collect and return
        loglik = np.dot(self.weights[self.Y0],loglik0[self.Y0])+np.dot(self.weights[self.Y1],loglik1[self.Y1])
        return sign*loglik

    def gradPoisson(self, parms, sign = 1.0):
        """
        Gradient of Zero Inflated Poisson Log-likelihood.
        """
        
        ## count mean
        eta = np.dot(self.X,parms[np.arange(self.kx)]) + self.offsetx
        mu = np.exp(eta)
        ## binary mean
        etaz = np.dot(self.Z, parms[np.arange(self.kx,self.kx+self.kz)]) + self.offsetz
        muz = self.linkobj.link_inv(etaz)
        ## densities at 0
        clogdens0 = -mu
        dens0 = muz*(1-self.Y1.astype(float)) + np.exp(np.log(1 - muz) + clogdens0)
        ## working residuals  
        wres_count = np.where(self.Y1,self.Y-mu,-np.exp(-np.log(dens0) + 
                                          np.log(1 - muz) + clogdens0 + np.log(mu))) 
        link_etaz = self.linkobj.link_inv_deriv(etaz)
        wres_zero  = np.where(self.Y1,-1/(1-muz) * link_etaz, \
                          (link_etaz - np.exp(clogdens0) * link_etaz)/dens0)   
    
        return sign*(np.hstack((np.expand_dims(wres_count*self.weights,axis=1)*self.X, \
                np.expand_dims(wres_zero*self.weights,axis=1)*self.Z))).sum(axis=0)
    
    def ziNegBin(self, parms, sign = 1.0):
        """
        Log-Likelihood of Zero Inflated Negative Binomial.
        """
        ## count mean
        mu = np.exp(np.dot(self.X,parms[np.arange(self.kx)]) + self.offsetx)
        ## binary mean
        phi = self.linkobj.link_inv(np.dot(self.Z, parms[np.arange(self.kx,self.kx+self.kz)]) + self.offsetz)
        ## negbin size
        theta = np.exp(parms[self.kx+self.kz])
    
        ## log-likelihood for y = 0 and y >= 1 sp.stats.poisson.logpmf(Y, mu)
        loglik0 = np.log(phi + np.exp(np.log(1-phi) + \
                                   st.nbinom.logpmf(0,*self.convert_params(theta = theta, mu = mu)) ) )
        loglik1 = np.log(1-phi) + st.nbinom.logpmf(self.Y,*self.convert_params(theta = theta, mu = mu))

        ## collect and return
        loglik = np.dot(self.weights[self.Y0],loglik0[self.Y0])+np.dot(self.weights[self.Y1],loglik1[self.Y1])
        return sign*loglik
  
    def ziGeom(self, parms, sign = 1.0):
        return self.ziNegBin(np.hstack((parms, 0)), sign)
    
    def gradGeom(self, parms, sign = 1.0):
        """
        Gradient of Zero Inflated Geometric Log-likelihood.
        
        """
        ## count mean
        eta = np.dot(self.X,parms[np.arange(self.kx)]) + self.offsetx
        mu = np.exp(eta)
        ## binary mean
        etaz = np.dot(self.Z, parms[np.arange(self.kx,self.kx+self.kz)]) + self.offsetz
        muz = self.linkobj.link_inv(etaz) 

        ## densities at 0
        clogdens0 = st.nbinom.logpmf(0,*self.convert_params(theta = 1, mu = mu))
        dens0 = muz*(1-self.Y1.astype(float)) + np.exp(np.log(1 - muz) + clogdens0)

        ## working residuals  
        wres_count = np.where(self.Y1,self.Y - mu*(self.Y + 1)/(mu + 1), \
                              -np.exp(-np.log(dens0) + np.log(1 - muz) + clogdens0 +\
                                      -np.log(mu+1) + np.log(mu))) 
        link_etaz = self.linkobj.link_inv_deriv(etaz)
        wres_zero  = np.where(self.Y1,-1/(1-muz) * link_etaz, \
                          (link_etaz - np.exp(clogdens0) * link_etaz)/dens0)
      
        return sign*(np.hstack((np.expand_dims(wres_count*self.weights,axis=1)*self.X, \
                np.expand_dims(wres_zero*self.weights,axis=1)*self.Z))).sum(axis=0)
    
    def gradNegBin(self, parms, sign = 1.0): 
        """
        Gradient of Zero Inflated Negative Binomial Log-likelihood. 
        (Negetive Binomial2 to be specific.)
        
        """
        ## count mean
        eta = np.dot(self.X,parms[np.arange(self.kx)]) + self.offsetx
        mu = np.exp(eta)
        ## binary mean
        etaz = np.dot(self.Z, parms[np.arange(self.kx,self.kx+self.kz)]) + self.offsetz
        muz = self.linkobj.link_inv(etaz)    
        ## negbin size
        theta = np.exp(parms[self.kx+self.kz])

        ## densities at 0
        clogdens0 = st.nbinom.logpmf(0,*self.convert_params(theta = theta, mu = mu))
        dens0 = muz*(1-self.Y1.astype(float)) + np.exp(np.log(1 - muz) + clogdens0)
        
        ## working residuals  
        wres_count = np.where(self.Y1,self.Y - mu*(self.Y + theta)/(mu + theta), \
                              -np.exp(-np.log(dens0) + np.log(1 - muz) + clogdens0 + np.log(theta) +\
                                      -np.log(mu+theta) + np.log(mu))) 
        link_etaz = self.linkobj.link_inv_deriv(etaz)
        wres_zero  = np.where(self.Y1,-1/(1-muz) * link_etaz, \
                          (link_etaz - np.exp(clogdens0) * link_etaz)/dens0)
        
        wres_theta = theta*np.where(self.Y1, sp.special.digamma(self.Y + theta) - sp.special.digamma(theta) +\
                                   np.log(theta) - np.log(mu + theta) + 1 - (self.Y + theta)/(mu + theta),\
                                   np.exp(-np.log(dens0) + np.log(1 - muz) + clogdens0)*\
                                   (np.log(theta) - np.log(mu + theta) + 1 - theta/(mu+theta) ) )
        
        return sign*(np.hstack((np.expand_dims(wres_count*self.weights,axis=1)*self.X, \
                np.expand_dims(wres_zero*self.weights,axis=1)*self.Z, \
                               np.expand_dims(wres_theta,axis=1)))).sum(axis=0)
    
    def EM_estimate(self):
        ## EM estimation of starting values
        
        model_count = sm.GLM(endog = self.Y, exog = self.X, family = sm.families.Poisson(),\
                                  offset = self.offsetx , freq_weights = self.weights).fit()
        model_zero = sm.GLM(self.Y0.astype(int), exog = self.Z, family=sm.families.Binomial(link = self.linkobj.linkclass), \
                   offset = self.offsetz , freq_weights = self.weights).fit()
        self.start = {'zero':model_zero.params, 'count':model_count.params}
        
        if self.dist is 'negbin':
            self.start['theta'] = 1.0 
            
        if (self.EM is True) and (self.dist is 'poisson'):
            mui = model_count.predict()
            probi = model_zero.predict()
            probi = probi/(probi + (1-probi)*sp.stats.poisson.pmf(0, mui))
            probi[self.Y1] = 0
            probi
            ll_new = self.loglikfun(np.hstack((self.start['count'].values,self.start['zero'].values)))
            ll_old = 2 * ll_new
    
            while np.absolute((ll_old - ll_new)/ll_old) > self.reltol :
                ll_old = ll_new
                model_count = sm.GLM(endog = self.Y, exog = self.X, family = sm.families.Poisson(),\
                                  offset = self.offsetx , freq_weights = self.weights*(1-probi) \
                                              ).fit(start_params = self.start['count'].values)        
                model_zero = sm.GLM(probi, exog = self.Z, family=sm.families.Binomial(link = self.linkobj.linkclass),\
                        offset = self.offsetz, freq_weights = self.weights \
                               ).fit(start_params = self.start['zero'].values)
                self.start = {'zero':model_zero.params, 'count':model_count.params}

                mui = model_count.predict()
                probi = model_zero.predict()
                probi = probi/(probi + (1-probi)*sp.stats.poisson.pmf(0, mui))
                probi[self.Y1] = 0

                ll_new = self.loglikfun(np.hstack((self.start['count'].values,self.start['zero'].values)))           
            
        if (self.EM is True) and (self.dist is 'geom'):
            mui = model_count.predict()
            probi = model_zero.predict()
            probi = probi/(probi + (1-probi)*st.nbinom.pmf(0,*self.convert_params(theta = 1, mu = mui)))
            probi[self.Y1] = 0
            
            ll_new = self.loglikfun(np.hstack((self.start['count'].values,self.start['zero'].values)))
            ll_old = 2 * ll_new  
                           
            while np.absolute((ll_old - ll_new)/ll_old) > self.reltol :
                ll_old = ll_new
                model_count = sm.GLM(endog = self.Y, exog = self.X, family = sm.families.NegativeBinomial(alpha = 1.0),\
                                  offset = self.offsetx , freq_weights = self.weights*(1-probi)).fit(\
                                        #start_params = start['count'].values
                                    sm.families.NegativeBinomial(alpha = 1.0\
                                                                ).starting_mu(y=self.start['count'].values))
                model_zero = sm.GLM(probi, exog = self.Z, family=sm.families.Binomial(link = self.linkobj.linkclass),\
                        offset = self.offsetz, freq_weights = self.weights).fit(start_params = self.start['zero'].values)
                self.start = {'zero':model_zero.params, 'count':model_count.params}

                mui = model_count.predict()
                probi = model_zero.predict()
                probi = probi/(probi + (1-probi)*st.nbinom.pmf(0,*self.convert_params(theta = 1, mu = mui)))
                probi[self.Y1] = 0                

                ll_new = self.loglikfun(np.hstack((self.start['count'].values,self.start['zero'].values)))
                
        if (self.EM is True) and (self.dist is 'negbin'):
            warnings.warn('EM estimation of starting values not optimal for Negetive Binomial.')
            mui = model_count.predict() # or model_count.mu
            probi = model_zero.predict()
            probi = probi/(probi + (1-probi)*st.nbinom.pmf(0,*self.convert_params(theta = self.start['theta'], mu = mui)))
            probi[self.Y1] = 0
            
            ll_new = self.loglikfun(np.hstack((self.start['count'].values,self.start['zero'].values,np.log(self.start['theta']))))
            ll_old = 2 * ll_new 
            
            while np.absolute((ll_old - ll_new)/ll_old) > self.reltol :
                ll_old = ll_new
                model_count = sm.GLM(endog = self.Y, exog = self.X, family = \
                                sm.families.NegativeBinomial(alpha = 1/self.start['theta']),method = 'newton',\
                                  offset = self.offsetx , freq_weights = self.weights*(1-probi) \
                                      ).fit(start_params = self.start['count'])
                model_zero = sm.GLM(probi, exog = self.Z, family=sm.families.Binomial(link = self.linkobj.linkclass),\
                        offset = self.offsetz, freq_weights = self.weights, \
                        start_params = self.start['zero']).fit()
                
                mui = model_count.predict()   
                theta = sm.GLM(endog = self.Y, exog = self.X, family = \
                                     sm.families.NegativeBinomial(alpha = 1/model_count.scale),method = 'newton',\
                                 offset = self.offsetx , freq_weights = self.weights*(1-probi) \
                                     ).estimate_scale(mui)
                
                probi = model_zero.predict()
                probi = probi/(probi + (1-probi)*st.nbinom.pmf(0,*self.convert_params(theta = theta, mu = mui)))
                
                probi[self.Y1] = 0
                self.start = {'zero':model_zero.params, 'count':model_count.params, 'theta':theta}
                
                ll_new = self.loglikfun(np.hstack((self.start['count'].values,\
                                self.start['zero'].values,np.log(self.start['theta']))))

        return self.start
    
     
    def fit(self, method = 'BFGS', EM = True, start = None, reltol = None,\
            options = {'disp': False, 'maxiter': 10000}, factr = 1.0):
        self.set_tolerance(factr, reltol)
        self.optim_options = options
        self.method = method
        self.EM = EM
        self.set_start(start)
        
        ## ML Estimation
        if (self.dist is 'negbin'):
            x0 = np.hstack((self.start['count'].values,self.start['zero'].values,\
                                         np.log(self.start['theta'])))
        else:
            x0 = np.hstack((self.start['count'].values,self.start['zero'].values))

        fitResult = sp.optimize.minimize(self.loglikfun, args=(-1.0,), x0 = x0, \
                                        method=self.method, jac=self.gradfun, options=self.optim_options)
        
        ## coefficients and covariances
        coefc = pd.Series(data = fitResult.x[0:self.kx], index = self.X.columns.values)
        coefz = pd.Series(data = fitResult.x[self.kx:self.kx+self.kz], index = self.Z.columns.values)

        if self.method == 'L-BFGS-B':
            vc_data = fitResult.hess_inv.todense()
        elif self.method == 'BFGS':
            vc_data = fitResult.hess_inv
        else:
            warnings.warn('Not tested for methods other than BFGS and L-BFGS-B.')
            
        vc = pd.DataFrame(data = vc_data[np.arange(self.kx+self.kz)[:,None],np.arange(self.kx+self.kz)], \
                      index = np.append(self.X.columns.values, self.Z.columns.values),\
                 columns = np.append(self.X.columns.values, self.Z.columns.values))
        if self.dist == 'negbin':
            ntheta = self.kx + self.kz
            theta = np.exp(fitResult.x[ntheta])
            SE_logtheta = np.sqrt(np.diagonal(vc_data)[ntheta])
        else:
            theta = None
            SE_logtheta = None
    
        ## fitted and residuals
        mu = np.exp(np.dot(self.X,coefc)+self.offsetx)
        phi = self.linkobj.link_inv(np.dot(self.Z,coefz)+self.offsetz)
        Yhat = (1-phi) * mu
        res = np.sqrt(self.weights) * (self.Y - Yhat)

        ## effective observations
        nobs = np.sum(self.weights > 0)
        
        Result = ZeroInflatedResults(self.call, self.formula, self.terms, self.kx, self.kz, \
                                     self.dist, self.link, self.linkobj, self.optim_options, self.method, self.start,\
                                     self.reltol, self.weights, self.offsetx, self.offsetz,\
                                     fitResult, coefc, coefz, theta, SE_logtheta, nobs, res, Yhat, vc, self.endog)

        return Result

        
        
    def set_start(self, start):
        if start is not None:
            valid = True
            if ('count' in start) is False:
                valid = False
                warnings.warn("invalid starting values, count model coefficients not specified")
                start['count'] = pd.Series(np.repeat(0,kx), index = X.columns.values)
            if ('zero' in start) is False:
                valid = False
                warnings.warn("invalid starting values, zero model coefficients not specified")
                start['zero'] = pd.Series(np.repeat(0,kz), index = Z.columns.values)
            if len(start['count']) != kx:
                valid = False
                warnings.warn("invalid starting values, wrong number of count model coefficients")
            if len(start['zero']) != kz:
                valid = False
                warnings.warn("invalid starting values, wrong number of zero model coefficients")
            if dist is 'negbin':
                if ('theta' in start) is False:
                    start['theta'] = 1.0
                start = {'zero':start['zero'], 'count':start['count'], 'theta' : (start['theta'][0]).astype(float)}
            else:
                start = {'zero':start['zero'], 'count':start['count']}    
        
            if valid is False:
                start = None

        if start is None:
            self.EM_estimate()
        else:
            self.start = start
        
     
    def set_tolerance(self, factr=1.0, reltol = FLOAT_EPS**(1/1.6)):
        if factr < 1.0:
            warnings.warn('Minimum value of factr is 1.0.')
            factr = 1.0
        if reltol is None:
            self.reltol = factr*(np.finfo(float).eps)**(1/1.6)
            
    @staticmethod    
    def formula_processing(formula_str, data, missing='none'):
        # ToDo: Add 'missing' operations on df
        X_formula,Z_formula = formula_str.split("|")
        Z_formula = X_formula.split("~")[0]+" ~ "+ Z_formula
        y, X = patsy.dmatrices(X_formula, data, return_type='dataframe')
        Z = patsy.dmatrices(Z_formula, data, return_type='dataframe')[1]
        
        Y = np.squeeze(y)
        ## sanity checks
        if len(Y) < 1:
            sys.exit("empty model")
        if np.all(Y > 0):
            sys.exit("invalid dependent variable, minimum count is not zero")  
        if np.array_equal(np.asarray(Y), (np.round(Y + 0.001)).astype(int)) is False:
            sys.exit("invalid dependent variable, non-integer values")
        Y = (np.round(y + 0.001)).astype(int)
        if np.any(Y < 0):
            sys.exit("invalid dependent variable, negative counts")
            
        return y,X,Z
    
    @staticmethod
    def convert_params(mu, theta):
        """
        Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

        """
        n = theta
        p = theta/(theta+mu)
        return n, p
            
    
    @staticmethod
    def _link_processing(link):
        ## binary link processing
        linkstr = link
        linkList = ['logit','probit','cauchit','cloglog','log']
        if linkstr not in linkList:
            warnings.warn(linkstr +" link not valid. Available links are: " + str(linkList))
            linkstr = 'logit'
        return(linkstr)
    
    
    @staticmethod
    def _LinkClass_processing(linkstr):
        Link = {
            'logit': Logit(),
            'probit': Probit(),
            'cloglog': CLogLog(),
            'cauchit': Cauchit(),
            'log': Log(),
        }
        return Link.get(linkstr, Logit())
    
    @staticmethod
    def _dist_processing(dist):
        if dist not in ['poisson','negbin','geom']:
            sys.exit(dist+" method not yet implemented")
        return dist
    

class ZeroInflatedResults(object):
    def __init__(self, call, formula, terms, kx, kz, dist, link, linkobj, options, method, start, reltol, \
                 weights, offsetx, offsetz, fitResult, coefc, coefz, theta, SE_logtheta,\
                nobs, res, Yhat, vc, y):
        
        # Need to change final results objects names to standard names used in python
        self.call = call
        self.formula = formula
        self.terms = terms
        self.kx = kx
        self.kz = kz
        self.n = self.nobs = nobs
        
        # fit paramters
        self.dist = dist
        self.linkstr = link
        self.link = linkobj.link
        self.linkinv = linkobj.link_inv
        self.optim_options = options
        self.method = method
        self.start = start
        self.reltol = reltol
                
        self.weights = weights 
        self.offsetx = offsetx
        self.offsetz = offsetz
        self.linkobj = linkobj
        
        # Optimization Results
        self.fit = fitResult
        self.loglik = fitResult.fun*(-1)    # log-likelihood
        self.converged = fitResult.success
        self.iters = fitResult.nit;          # number of iterations for convergence
        self.coefficients = {'count':coefc ,'zero': coefz}
        self.theta = theta if (dist is 'negbin') else None
        self.SE_logtheta = SE_logtheta
        self.df_null = nobs - 2
        self.df_resid = nobs - (kx + kz + (dist == "negbin"))
        self.df_model = (kx + kz + (dist == "negbin"))
        self.residuals = res 
        self.fitted_values = Yhat
        self.vcov = vc
        self.y = y
        
        self.deviance = 0
        self.pearson_chi2 = 0
        self.cov_type = None
        self.use_t = False
        
    def __repr__(self):
        display_string = "Call:\n    "+self.call
        display_string += "\n\nformula:\n    "+self.formula
        #display_string += f"\nterms:\n    Y: {self.terms['Y']}\n    X: {self.terms['X']}\n    Z: {self.terms['Z']}"
        display_string += "\ndist: "+self.dist
        display_string += "\nlink: "+self.linkstr
        display_string += "\nlinkobj:"+ self.linkobj.__repr__()
        display_string += f"\nMessage: {self.fit.message}"
        display_string += f"\nResult: \n    Count:\n"
        display_string += f"{self.coefficients['count']}\n    Zero:\n{self.coefficients['zero']}"
        display_string += f"\n    theta: {self.theta:0.12f}" if self.dist is 'negbin' else " "
        display_string += f"\ndf_null: {self.df_null} \ndf_resid: {self.df_resid}"
        return display_string
    
    # Merged function definitions from zipModel start here:
    def printModel(self):
        MODEL1_HEADER = f"Count model cefficients ({self.dist} log link): "
        MODEL2_HEADER = f"Zero-inflation model coefficients (binomial with {self.linkstr} link): "
    
        p_count = self.kx;   # num of preds (+ intercept) for poisson count
        p_zero  = self.kz;    # num of preds (+ intercept) for zero-inflated model
        
        # print string representation of model
        print("\nCall:\n" + self.call + "\n")
    
        # part 1: poisson count model
        print(MODEL1_HEADER + "\n")
        for line in [self.terms['X'], np.round(self.coefficients['count'],4)]:
            print(('{:>12}' * p_count).format(*line))
        print("\n")
        
        # part 2: logit model for predicting excess zeros
        print(MODEL2_HEADER + "\n")
        for line in [self.terms['Z'], np.round(self.coefficients['zero'],4)]:
            print(('{:>12}' * p_zero).format(*line))
        print("\n")
    
    def covar(self):
        print('definition here')        
        
    def summary(self):
        RESIDUAL_OUTPUT = "Pearson residuals:"
        MODEL1_HEADER = f"Count model cefficients ({self.dist} log link): "
        MODEL2_HEADER = f"Zero-inflation model coefficients (binomial with {self.linkstr} link): "

        
        ## chunk 1: output call, formula
        print("\nCall:\n" + self.call + "\n")
        
        
        ## chunk 2: output pearson residuals -- residuals function still needs to be implemented
        # object$residuals = residuals(object, type = "pearson")
        resid_summ = np.round(st.mstats.mquantiles(self.residuals, prob = [0, 0.25, 0.5, 0.75, 1.0]), 5)
        resid_str  = ['Min', '1Q', 'Median', '3Q','Max'];
        print(RESIDUAL_OUTPUT + '\n')
        for line in [resid_str, resid_summ]:
            print(('{:>10}' * len(resid_summ)).format(*line))
        print("\n")

        # compute standard error for all coefficients (both models)
        se = np.sqrt(np.diagonal(self.vcov)) 

        # compute z statistics for both models
        z_count = np.array(self.coefficients['count']) / se[0:self.kx]
        z_zip = np.array(self.coefficients['zero']) / se[self.kx:self.kx+self.kz]

        # compute p-values
        pval_count = 2 * st.norm.cdf(-np.abs(z_count));
        pval_zip = 2 * st.norm.cdf(-np.abs(z_zip));
        
        # format p-values for output
        pc_format = [0] * len(pval_count)
        pz_format = [0] * len(pval_zip)

        for i in np.arange(len(pval_count)):
            if pval_count[i] < 2e-16:
                pc_format[i] = str("<2e-16")
            else:
                pc_format[i] = str("{:.4e}".format(Decimal(pval_count[i])))

        for i in np.arange(len(pval_zip)):
            if pval_zip[i] < 2e-16:
                pz_format[i] = str("<2e-16")
            else:
                pz_format[i] = str("{:.4e}".format(Decimal(pval_zip[i])))
                
        ## chunk 3: output count model coefficients
        print(MODEL1_HEADER + "\n")
        coeff_label = ['', 'Estimate', 'Std. Error', 'z value', 'Pr(>|z|)'];
        data_count = [coeff_label] + list(zip(self.terms['X'], np.round(self.coefficients['count'],4), \
                                        np.round(se[0:self.kx], 4), np.round(z_count, 3), pc_format))
        coeff_maxlen = np.max([len(x) for x in self.terms['X']])
        SPACE = 20 if coeff_maxlen > 10 else 12
        
        for i, d in enumerate(data_count):
            line = '|'.join(str(x).rjust(SPACE) for x in d)
            print(line)
            if i == 0:
                print('-' * len(line))

        print('\n')
                    
        ## chunk 4: output zero-inflation model coefficients
        print(MODEL2_HEADER + "\n")
        data_zero = [coeff_label] + list(zip(self.terms['Z'], np.round(self.coefficients['zero'],4), \
                                        np.round(se[self.kx:], 4), np.round(z_zip, 3), pz_format))
        coeff_maxlen = np.max([len(x) for x in self.terms['Z']])
        SPACE = 20 if coeff_maxlen > 10 else 12  
        for i, d in enumerate(data_zero):
            line = '|'.join(str(x).rjust(SPACE) for x in d)
            print(line)
            if i == 0:
                print('-' * len(line))
        print('\n')
        print('---')
        
        ## chunk 5: Number of iterations, log-likelihood
        print(f"Number of iterations in {self.method} optimization: " + str(self.iters));
        if self.converged is False:
            print("Failed to converge.")
        print("Log-likelihood: " + str(self.loglik))        