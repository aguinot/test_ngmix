import numpy as np
from numpy.random import uniform as urand

import galsim

import ngmix

from joblib import Parallel, delayed
from tqdm import tqdm


# np.random.seed(1234)



class Sersic():
    """
    """

    def __init__(self):
        """
        """

        # Set numpy and galsim seeds
        self._numpy_rand = np.random.seed(42)
        self.galsim_seed = galsim.UniformDeviate(42)


    def _create_sersic(self, g1=0., g2=0.):
        """Create Sersic
        """

        n = 1.2  # Sersic index
        hlr=0.5  # Half light radius arcsec
        flux=100.  # Flux (doesn't really matter since we prefer to fix the SNR)
        g1_i = 0.02
        g2_i = 0.
        gal = galsim.Sersic(n=n, half_light_radius=hlr, flux=flux)  # Create Sersic profile
        # gal = galsim.Exponential(half_light_radius=hlr, flux=flux)  #  Create exponential profile
        gal = gal.shear(g1=g1_i, g2=g2_i)

        return gal, np.array([g1_i, g2_i])

    
    def _create_psf(self, g1, g2, fwhm=0.65):
        """Create Moffat
        """

        beta = 4.765  # Value from https://arxiv.org/pdf/astro-ph/0109067.pdf

        psf = galsim.Moffat(beta=beta, fwhm=fwhm).withFlux(1)

        psf = psf.shear(g1=g1, g2=g2)

        return psf


    def _get_one_gal(self, gal_g1=0.05, gal_g2=-0.03, psf_fwhm=0.65, psf_g1=-0.01, psf_g2=0.01, img_shape=(51,51), img_pixel_scale=0.186, sigma_noise=None, snr=None, rng=None):
        """Create one object
        """

        # This allow crontrol of random over parallel processes
        galsim_rng = galsim.UniformDeviate(int(rng.rand()*100000))

        error = 1
        while error:
            try:
                gal, g = self._create_sersic(gal_g1, gal_g2)

                psf = self._create_psf(psf_g1, psf_g2, psf_fwhm)

                obj = galsim.Convolve(gal, psf)

                obj_galimg = obj.drawImage(nx=img_shape[0], ny=img_shape[1], scale= img_pixel_scale)

                psf_galimg = psf.drawImage(nx=img_shape[0], ny=img_shape[1], scale= img_pixel_scale)

                if sigma_noise != None:
                    obj_galimg.addNoise(galsim.GaussianNoise(sigma=sigma_noise, rng=galsim_rng))
                elif snr != None:
                    obj_galimg.addNoiseSNR(noise=galsim.GaussianNoise(sigma=0.01, rng=galsim_rng), snr=snr)

                error = 0
            except:
                print('error')
                error = 1

        return obj_galimg.array, psf_galimg.array, g


    def get_gal(self, gal_g1=0.05, gal_g2=-0.03, psf_fwhm=0.65, psf_g1=-0.01, psf_g2=0.01, n_gal=1, img_shape=(51,51), img_pixel_scale=0.186, sigma_noise=None, snr=None):
        """
        """
        # This allow crontrol of random over parallel processes
        seeds = (np.random.rand(n_gal)*100000).astype(int)

        print('Make simu...')
        res = (Parallel(n_jobs=-1, backend=None)
               (delayed(self._get_one_gal)
               (gal_g1, gal_g2, 
                psf_fwhm, psf_g1, psf_g2, 
                img_shape, img_pixel_scale, 
                sigma_noise, snr, np.random.RandomState(seed)) for seed in tqdm(seeds, total=len(seeds))))

        final_gal, final_psf, all_g = zip(*res)
        final_gal = np.array(final_gal)
        final_psf = np.array(final_psf)
        all_g = np.array(all_g)

        return final_gal, final_psf, all_g, seeds


class Shapes():
    """
    """

    def __init__(self, gal_array, psf_array, true_g, pixel_scale=0.186, make_ngmix=True, make_galsim=True, make_galsim_mcal=True, galsim_type='', rng_seed=42):

        self._gal_array = gal_array
        self._psf_array = psf_array
        self._true_g = true_g
        self._pixel_scale = pixel_scale
        self._make_ngmix = make_ngmix
        self._make_galsim = make_galsim
        if self._make_galsim:
            self._galsim_type = galsim_type
        self._make_galsim_mcal = True

        np.random.seed(rng_seed)

    @staticmethod
    def mad(data, axis=None):
        """Median absolute deviation
        Compute the mad of an array.
        Parameters
        ----------
        data : numpy.ndarray
            Data array
        axis : int, tuple
            Axis on which computing the mad.
        """
        return np.median(np.abs(data - np.median(data, axis)), axis)*1.4826
    

    def _get_shape_ngmix(self, gal_img, psf_img, psf_pars=[0.0, 0.0, -0.01, 0.01, 0.15, 1.0], seed=None):
        """
        """

        np.random.seed(seed)

        obs = self._get_ngmix_obs(gal_img, psf_img)

        res = self._run_metacal(obs)

        # Compile the results
        output = []
        error = False
        if (res['mcal_flags'] == 0):
            for key in ['1m', '1p', '2m', '2p', 'noshear']:
                if res[key]['flags'] != 0:
                    error = True
                    break
                output.append(res[key]['g'])
        else:
            error = True

        if error:
            output = [np.array([-10., -10.]) for i in range(5)]

        return output


    def _run_metacal(self, obs):
        """
        """

        boot = ngmix.bootstrap.MaxMetacalBootstrapper(obs)

        psf_model = 'em3'
        gal_model = 'gauss'

        prior = self._get_prior()

        metacal_pars = {'types': ['noshear', '1p', '1m', '2p', '2m'],
                        'psf': 'gauss',
                        'fixnoise': True,
                        'cheatnoise': False,
                        'symmetrize_psf': False}
        
        lm_pars = {'maxfev': 2000,
                   'xtol': 5.0e-5,
                   'ftol': 5.0e-5}

        max_pars = {'method': 'lm',
                    'lm_pars': lm_pars}

        psf_pars = {'maxiter': 5000,
                    'tol': 5.0e-6}

        Tguess = 0.15
        ntry = 5
        boot.fit_metacal(psf_model,
                         gal_model,
                         max_pars,
                         Tguess,
                         prior=prior,
                         ntry=ntry,
                         metacal_pars=metacal_pars,
                         psf_fit_pars=psf_pars,
                         psf_ntry=20)

        res = boot.get_metacal_result()

        return res


    def _get_shape_galsim(self, gal_img, psf_img, galsim_type=''):
        """
        """

        if galsim_type == '':
            s = galsim.hsm.EstimateShear(galsim.Image(gal_img, scale=self._pixel_scale), galsim.Image(psf_img, scale=self._pixel_scale), strict=False)
            try:
                ss = galsim.Shear(e1=s.corrected_e1, e2=s.corrected_e2)
                g_gal = np.array([ss.g1, ss.g2])
                g_psf = np.array([s.psf_shape.g1, s.psf_shape.g2])
            except:
                g_gal = np.array([-10, -10])
                g_psf = np.array([-10, -10])

        elif galsim_type == 'ksb':
            s = galsim.hsm.EstimateShear(galsim.Image(gal_img, scale=self._pixel_scale), galsim.Image(psf_img, scale=self._pixel_scale), shear_est='KSB', strict=False)
            try:
                ss = galsim.Shear(g1=s.corrected_e1, g2=s.corrected_e2)
                g_gal = np.array([ss.g1, ss.g2])
                g_psf = np.array([s.psf_shape.g1, s.psf_shape.g2])
            except:
                g_gal = np.array([-10, -10])
                g_psf = np.array([-10, -10])

        return g_gal, g_psf


    def _get_galsim_metacal(self, gal_img, psf_img, galsim_type='', seed=None):
        """
        """

        np.random.seed(seed)

        obs = self._get_ngmix_obs(gal_img, psf_img)

        obs_metacal = ngmix.metacal.get_all_metacal(obs,
                                                    type=['noshear', '1p', '1m', '2p', '2m'],
                                                    fixnoise=True,
                                                    cheatnoise=False,
                                                    step=0.01,
                                                    psf='gauss')

        output = []
        for key in ['1m', '1p', '2m', '2p', 'noshear']:
            psf_tmp_galobj = obs_metacal[key].get_psf().galsim_obj
            psf_tmp = psf_tmp_galobj.drawImage(nx=gal_img.shape[0], ny=gal_img.shape[1], scale=self._pixel_scale).array
            gal_tmp = obs_metacal[key].image
            res_tmp = self._get_shape_galsim(gal_tmp, psf_tmp)
            output.append(res_tmp[0])

        return output


    def _get_ngmix_obs(self, gal_img, psf_img, psf_pars=[0.0, 0.0, -0.01, 0.01, 0.15, 1.0]):
        """
        """

        eps=0.01
        lm_pars={'maxfev': 2000,
                 'xtol': 5.0e-5,
                 'ftol': 5.0e-5}

        img_shape = gal_img.shape

        # Jacobian
        gal_jacob=ngmix.DiagonalJacobian(scale=self._pixel_scale, x=int((img_shape[0]-1)/2.), y=int((img_shape[1]-1)/2.))
        psf_jacob=ngmix.DiagonalJacobian(scale=self._pixel_scale, x=int((img_shape[0]-1)/2.), y=int((img_shape[1]-1)/2.))

        # PSF fittitng
        psf_noise = np.sqrt(np.sum(psf_img**2)) / 500
        psf_weight = np.ones_like(psf_img) / psf_noise**2
        psf_obs=ngmix.Observation(psf_img, weight=psf_weight, jacobian=psf_jacob)

        pfitter=ngmix.fitting.LMSimple(psf_obs,'gauss',lm_pars=lm_pars)

        guess=np.array(psf_pars)
        guess[0] += urand(low=-eps,high=eps)
        guess[1] += urand(low=-eps,high=eps)
        guess[2] += urand(low=-eps, high=eps)
        guess[3] += urand(low=-eps, high=eps)
        guess[4] *= (1.0 + urand(low=-eps, high=eps))
        guess[5] *= (1.0 + urand(low=-eps, high=eps))

        pfitter.go(guess)

        # print(np.abs((pfitter.get_result()['g']-np.array([-0.01, 0.01]))/np.array([-0.01, 0.01])))

        psf_gmix_fit = pfitter.get_gmix()

        psf_obs.set_gmix(psf_gmix_fit)

        # Gal fitting
        # Get noise level direclty from the image
        sigma_noise = self.mad(gal_img)
        noise = np.random.normal(size=img_shape) * sigma_noise
        weight = np.ones_like(gal_img) * 1/sigma_noise**2.

        obs = ngmix.Observation(gal_img, weight=weight, noise=noise, jacobian=gal_jacob, psf=psf_obs)

        return obs


    def _get_prior(self):
        """ Get prior
    ​
        Return prior for the different parameters
        (Ngmix repo wiki: https://github.com/esheldon/ngmix/wiki/Metacalibration)
    ​
        Return
        ------
        prior : ngmix.priors
            Priors for the different parameters.
    ​
        """

        # prior on ellipticity.  The details don't matter, as long
        # as it regularizes the fit.  This one is from Bernstein & Armstrong 2014
        g_sigma = 0.4
        g_prior = ngmix.priors.GPriorBA(g_sigma)

        # 2-d gaussian prior on the center
        # row and column center (relative to the center of the jacobian, which
        # would be zero)
        # and the sigma of the gaussians
        # units same as jacobian, probably arcsec
        row, col = 0.0, 0.0
        row_sigma, col_sigma = self._pixel_scale, self._pixel_scale  # pixel size
        cen_prior = ngmix.priors.CenPrior(row, col, row_sigma, col_sigma)
    
        # T prior.  This one is flat, but another uninformative you might
        # try is the two-sided error function (TwoSidedErf)
        Tminval = -10.0  # arcsec squared
        Tmaxval = 1.e6
        T_prior = ngmix.priors.FlatPrior(Tminval, Tmaxval)
    
        # similar for flux.  Make sure the bounds make sense for
        # your images
        Fminval = -1.e4
        Fmaxval = 1.e9
        F_prior = ngmix.priors.FlatPrior(Fminval, Fmaxval)
    
        # now make a joint prior.  This one takes priors
        # for each parameter separately
        prior = ngmix.joint_prior.PriorSimpleSep(cen_prior,
                                                g_prior,
                                                T_prior,
                                                F_prior)
    
        return prior


    def get_shapes(self):
        """
        """

        final = {}
        if self._make_galsim:
            print("Running galsim...")

            res_galsim = (Parallel(n_jobs=-1, backend=None)
                          (delayed(self._get_shape_galsim)
                          (gal, psf, self._galsim_type) for gal, psf in tqdm(zip(self._gal_array, self._psf_array), total=len(self._gal_array))))
            res_tmp = list((zip(*res_galsim)))
            final['galsim'] = {'gal': np.array(res_tmp[0]), 'psf': np.array(res_tmp[1])}

        if self._make_galsim_mcal:
            print("Running galsim + metacal...")

            seeds = (np.random.rand(len(self._gal_array))*100000).astype(int)

            res_galsim_mcal = (Parallel(n_jobs=-1, backend=None)
                               (delayed(self._get_galsim_metacal)
                               (gal, psf, seed=seed) for gal, psf, seed in tqdm(zip(self._gal_array, self._psf_array, seeds), total=len(self._gal_array))))
            res_tmp = list((zip(*res_galsim_mcal)))
            keys = ['1m', '1p', '2m', '2p', 'noshear'] 
            final['galsim_mcal'] = {key: np.array(res_tmp[i]) for i, key in enumerate(keys)}

        if self._make_ngmix:
            print("Running ngmix...")

            seeds = (np.random.rand(len(self._gal_array))*100000).astype(int)

            res_ngmix = (Parallel(n_jobs=-1, backend=None)
                         (delayed(self._get_shape_ngmix)
                         (gal, psf, seed=seed) for gal, psf, seed in tqdm(zip(self._gal_array, self._psf_array, seeds), total=len(self._gal_array))))
            res_tmp = list((zip(*res_ngmix)))
            keys = ['1m', '1p', '2m', '2p', 'noshear']
            final['ngmix'] = {key: np.array(res_tmp[i]) for i, key in enumerate(keys)}
            # final['ngmix'] = res_ngmix

        self.final = final
    
    def get_err_shapes(self):
        """
        Compute the errors onn shape measurement
        """

        if self._make_galsim:
            bad_flags = np.where(self.final['galsim']['gal'][:,0] != -10)[0]
            err = (self.final['galsim']['gal'][bad_flags] - self._true_g.squeeze())/1e-3
            print()
            print('Galsim error')
            print('------------')
            print("Galaxy rejected because of errors : {}".format(len(self.final['galsim']['gal'][:,0]) - len(self.final['galsim']['gal'][:,0][bad_flags])))
            print('e1 = {}      e2 = {}    10^-3'.format(*np.mean(err[bad_flags], 0)))

        if self._make_galsim_mcal:
            bad_flags = self.final['galsim_mcal']['noshear'][:,0] != -10
            for key in ['1m', '1p', '2m', '2p']:
                bad_flags &= self.final['galsim_mcal'][key][:,0] != -10
            R11 = (self.final['galsim_mcal']['1p'][:,0] - self.final['galsim_mcal']['1m'][:,0])/2./0.01 
            R22 = (self.final['galsim_mcal']['2p'][:,1] - self.final['galsim_mcal']['2m'][:,1])/2./0.01
            g1 = self.final['galsim_mcal']['noshear'][:,0]
            g2 = self.final['galsim_mcal']['noshear'][:,1]

            m, merr, c, cerr = self._jack_est(g1[bad_flags], R11[bad_flags] * self._true_g[:,0][bad_flags], g2[bad_flags], R22[bad_flags])
            
            err = (np.array([g1[bad_flags], g2[bad_flags]]).T/np.mean([R11[bad_flags], R22[bad_flags]], 1) - self._true_g.squeeze()[bad_flags])/1e-3
            print()
            print('Galsim + Metacal error')
            print('----------------------')
            print("Galaxy rejected because of errors : {}".format(len(g1) - len(g1[bad_flags])))
            print('e1 = {}      e2 = {}    10^-3'.format(*np.mean(err[bad_flags], 0)))
            print("m [1e-3] : {m:f} +/- {msd:f}\nc [1e-4] : {c:f} +/- {csd:f}\nm : {m2:f} +/- {msd2:f}\nc : {c2:f} +/- {csd2:f}".format(
                        m=m/1e-3,
                        msd=merr/1e-3,
                        c=(c-np.mean(self._true_g[:,1]))/1e-4,
                        csd=cerr/1e-4,
                        m2=m,
                        msd2=merr,
                        c2=c-np.mean(self._true_g[:,1]),
                        csd2=cerr), flush=True)
        
        if self._make_ngmix:
            bad_flags = self.final['ngmix']['noshear'][:,0] != -10
            for key in ['1m', '1p', '2m', '2p']:
                bad_flags &= self.final['ngmix'][key][:,0] != -10
            R11 = (self.final['ngmix']['1p'][:,0] - self.final['ngmix']['1m'][:,0])/2./0.01 
            R22 = (self.final['ngmix']['2p'][:,1] - self.final['ngmix']['2m'][:,1])/2./0.01
            self._bad_flags = bad_flags
            g1 = self.final['ngmix']['noshear'][:,0]
            g2 = self.final['ngmix']['noshear'][:,1]

            m, merr, c, cerr = self._jack_est(g1[bad_flags], R11[bad_flags] * self._true_g[:,0], g2[bad_flags], R22[bad_flags])
            
            err = (np.array([g1, g2]).T/np.mean([R11, R22], 1) - self._true_g.squeeze())/1e-3
            print()
            print('Ngmix error')
            print('-----------')
            print("Galaxy rejected because of errors : {}".format(len(g1) - len(g1[bad_flags])))
            print('e1 = {}      e2 = {}    10^-3'.format(*np.mean(err[bad_flags], 0)))
            print("m [1e-3] : {m:f} +/- {msd:f}\nc [1e-4] : {c:f} +/- {csd:f}\nm : {m2:f} +/- {msd2:f}\nc : {c2:f} +/- {csd2:f}".format(
                        m=m/1e-3,
                        msd=merr/1e-3,
                        c=(c-np.mean(self._true_g[:,1]))/1e-4,
                        csd=cerr/1e-4,
                        m2=m,
                        msd2=merr,
                        c2=c-np.mean(self._true_g[:,1]),
                        csd2=cerr), flush=True)


    def _jack_est(self, g1, R11, g2, R22):
        """
        Code from Becker in : https://github.com/esheldon/ngmix/issues/72
        """
        g1bar = np.mean(g1)
        R11bar = np.mean(R11)
        g2bar = np.mean(g2)
        R22bar = np.mean(R22)
        n = g1.shape[0]
        fac = n / (n-1)
        m_samps = np.zeros_like(g1)
        c_samps = np.zeros_like(g1)

        for i in range(n):
            _g1 = fac * (g1bar - g1[i]/n)
            _R11 = fac * (R11bar - R11[i]/n)
            _g2 = fac * (g2bar - g2[i]/n)
            _R22 = fac * (R22bar - R22[i]/n)
            m_samps[i] = _g1 / _R11 - 1
            c_samps[i] = _g2 / _R22

        m = np.mean(m_samps)
        c = np.mean(c_samps)

        m_err = np.sqrt(np.sum((m - m_samps)**2) / fac)
        c_err = np.sqrt(np.sum((c - c_samps)**2) / fac)

        return m, m_err, c, c_err


cgal = Sersic()
r = cgal.get_gal(n_gal=5000, snr=30, img_pixel_scale=0.2, img_shape=(101, 101))

obs_shape = Shapes(r[0], r[1], r[2], pixel_scale=0.2)
obs_shape.get_shapes()
obs_shape.get_err_shapes()
