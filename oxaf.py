# -*- coding: utf-8 -*-
"""
=============================================================================

OXAF
A physically-based model of the ionizing radiation from active galaxies for
photoionization modeling
Adam D. Thomas 2016
adam.thomas@anu.edu.au

=============================================================================

REFERENCE
A. D. Thomas et al. 2016, ApJ, in prep.  Submitted August 2016.
Draft abstract reproduced below:

A physically-based model of the ionizing radiation from active galaxies for
photoionization modeling

We present a new model of Active Galactic Nucleus (AGN) continuum emission
designed for photoionization modeling.  The new model {\sc oxaf} reproduces the
diversity of spectral shapes that arise in physically-based models.  We
identify and explain degeneracies in the effects of AGN parameters on model
spectral shapes, with a focus on the complete degeneracy between the black hole
mass and AGN luminosity.  Our re-parametrized model {\sc oxaf} removes these
degeneracies and accepts three parameters which directly describe the output
spectral shape: the energy of the peak of the accretion disk emission
$E_{\mathrm{peak}}$, the photon power-law index of the non-thermal emission
$\Gamma$, and the proportion of the total flux which is emitted in the non-
thermal component $p_{\mathrm{NT}}$. The parameter $E_{\mathrm{peak}}$ is
presented as a function of the black hole mass, AGN luminosity, and `coronal
radius' of the {\sc optxagnf} model upon which {\sc oxaf} is based.  We show
that the soft X-ray excess does not significantly affect photoionization
modeling predictions of strong emission lines in Seyfert narrow-line regions.
Despite its simplicity, {\sc oxaf} accounts for opacity effects where the
accretion disk is ionized because it inherits the `color correction' of
{\sc optxagnf}. We use a grid of {\sc mappings} photoionization models with
{\sc oxaf} ionizing spectra to demonstrate how predicted emission-line ratios
on standard optical diagnostic diagrams are sensitive to each of the three
{\sc oxaf} parameters.  The {\sc oxaf} code is publicly available in the
Astrophysics Source Code Library.

=============================================================================

USAGE
This module may be used in two ways:
 1) Run as a command-line program to output the model spectrum to stdout:
        python oxaf.py E_peak Gamma p_NT
    Enter an incorrect number of command-line parameters to see the help string.
 2) Import into python (interactively or in a script) to gain access to the
    functions.  For example:
        import oxaf
        E, B, F = oxaf.full_spectrum(E_peak, Gamma, p_NT)
    Documentation for each of the functions is given in the function docstring.
    E.g. in ipython, "oxaf.full_spectrum?" will display help for this function.

The module may be used in python 2 or python 3, and depends only on the
standard numerical library numpy.

The module contains the following non-private functions:
find_E_peak        Calculate E_peak from M_BH, L/L_Edd and r_cor
disk               Calculate the oxaf model accretion disk spectrum      
non_thermal        Calculate the oxaf model non-thermal power-law spectrum 
full_spectrum      Calculate the full oxaf model spectrum: call "disk" and
                   "non_thermal" and sum the results using the weighting p_NT

See individual functions for further documentation.

=============================================================================
"""



# These "__future__" imports allow using both Python 2 and Python 3
from __future__ import division, print_function
import numpy as np  # Standard python numerical package

# Global variables:
__version__ = 2.1
E_norm_range = (0.01, 20)  # keV.  Spectra will be normalized over this range.
output_range = (1e-4, 500) # keV   



def find_E_peak(L, M, r):
    """
    Find the energy of the peak of the Big Blue Bump (BBB) disk emission E_peak,
    given the three relevant (but fully or partially degenerate) optxagnf
    parameters:
        L: The AGN luminosity in log10(L/L_Edd)
        M: The black hole mass in log10(M/M_Sun)
        r: The "coronal radius" in log10(r/r_g)
    Note that L_Edd and r_g depend on M.
    Returns an estimate for the energy of the peak of the BBB in log10(keV).
    Note that the "peak" refers to a plot of log10(keV*((keV/cm2/s)/keV)) 
    vs log10(E/keV), i.e. log10(E*F_E) for each bin versus
    log10(bin energy in keV).
    This function implements Equation 4 in the paper, using the fit parameters
    in Table 1.
    """
    if not (0.77815 <= r <= 3):
        raise ValueError("r (log10(r/rg)) must be between 0.77815 and 3")

    # Define constants:
    A_x2, A_x1, A_x0 = (-0.18051, -0.00812, 0.58599)
    B_c =  0.2501
    b3_x1, b3_x0 = (0.03426, -0.0187)
    b1_x2, b1_x1, b1_x0 = ( 0.39141, -0.83224,  0.60778)
    a__x2, a__x1, a__x0 = (-0.29700,  0.39399, -0.82247)

    return max( ( (A_x2*r*r + A_x1*r + A_x0) + B_c*(L - M) ),
                (             (b3_x1*r + b3_x0)*(L - M + 6.0)**3 + 
                  (b1_x2*r*r + b1_x1*r + b1_x0)*(L - M + 6.0)    + 
                  (a__x2*r*r + a__x1*r + a__x0)                    ) )



def _find_E_vec(E_peak):
    """
    Determine a vector of energy bins, ensuring that it covers the vast 
    majority of the flux for both the BBB disk component and the power-law
    component (i.e. we may use a larger range internally than the output range).
        E_peak: The energy of the peak of the BBB, in log10(E/keV).
                The peak is in a plot of log10(keV*((keV/cm2/s)/keV)) vs
                log10(E/keV), i.e. log10(E*F_E) vs log10(bin energy in keV).
    Returns both a vector of bin energies (keV; the bin centres in log(E/keV)
    space), and also a vector of bin widths (keV; the bins have equal widths
    in log(E) space, not linear space).
    """
    o_min, o_max = output_range # keV
    o_n = 3000 # number of bins
    D_lg10_E = np.log10(o_max / o_min) / o_n  # Constant bin width in dex(keV)

    # Determine minimum and maximum of our bin range:
    # (EAR = "Energy At Right")
    # Ensure we start (end) at least 4 (3) dex below (above) E_peak
    min_EAR = min(10**E_peak / 1e4, o_min)  # keV
    max_EAR = max(o_max, 10**E_peak * 1e3)  # keV
    
    # Number of bins below the bottom of the output range:
    n_below = np.floor(np.log10(o_min/min_EAR) / D_lg10_E)
    # Number of bins above the top of the output range:
    n_above = np.floor(np.log10(max_EAR/o_max) / D_lg10_E)

    # Calculate vector of energies (right edges of bins; log10(keV)):
    lg10_EAR = np.arange(np.log10(o_min) -  n_below*D_lg10_E,
                         np.log10(o_max) + (n_above+1)*D_lg10_E, D_lg10_E)
    # Calculate vector of bin widths (Has one less element than lg10_EAR)
    bin_widths = 10**(lg10_EAR[1:]) - 10**(lg10_EAR[:-1]) # keV.  
    # Calculate vector of energies at the centres of the bins:
    # Has one less element than lg10_EAR.
    # Bin centres match calculated bin widths
    EAC = 10**( lg10_EAR[1:] - D_lg10_E/2.0 )  # keV.  "Energy At Centre"
    # Note that EAC gives the bin centres in log space, not linear space.

    return EAC, bin_widths  # log10(keV), keV



def _donthcomp(ear, param):
    """
    This function was adapted by ADT from the subroutine donthcomp in
    donthcomp.f, distributed with XSpec.
    Nthcomp documentation:
    https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSmodelNthcomp.html
    Refs:
    Zdziarski, Johnson & Magdziarz 1996, MNRAS, 283, 193,
    as extended by Zycki, Done & Smith 1999, MNRAS 309, 561

    Note that the subroutine has been modified so that parameter 4
    is ignored, and the seed spectrum is always a blackbody.

    ear: Energy vector, listing "Energy At Right" of bins (keV)
    param: list of parameters; see the 5 parameters listed below.

    The original fortran documentation for this subroutine is included below:

    Driver for the Comptonization code solving Kompaneets equation
    seed photons  -  (disk) blackbody
    reflection + Fe line with smearing
    
    Model parameters:
    1: photon spectral index
    2: plasma temperature in keV
    3: (disk)blackbody temperature in keV
    4: type of seed spectrum (0 - blackbody, 1 - diskbb)
    5: redshift
    """
    ne = ear.size  # Length of energy bin vector
    # Note that this model does not calculate errors.
    #c     xth is the energy array (units m_e c^2)
    #c     spnth is the nonthermal spectrum alone (E F_E)
    #c     sptot is the total spectrum array (E F_E), = spref if no reflection
    zfactor = 1.0 + param[5]
    #c  calculate internal source spectrum
    #                           blackbody temp,   plasma temp,      Gamma
    xth, nth, spt = _thcompton(param[3] / 511.0, param[2] / 511.0, param[1])
    # The temperatures are normalized by 511 keV, the electron rest energy
    # Calculate normfac:
    xninv = 511.0 / zfactor
    ih = 1
    xx = 1.0 / xninv
    while (ih < nth and xx > xth[ih]):
        ih = ih + 1
    il = ih - 1
    spp = spt[il] + (spt[ih] - spt[il]) * (xx - xth[il]) / (xth[ih] - xth[il])
    normfac = 1.0 / spp

    #c     zero arrays
    photar = np.zeros(ne)
    prim   = np.zeros(ne)
    #c     put primary into final array only if scale >= 0.
    j = 0
    for i in range(0, ne):
        while (j <= nth and 511.0 * xth[j] < ear[i] * zfactor):
            j = j + 1
        if (j <= nth):
            if (j > 0):
                jl = j - 1
                prim[i] = spt[jl] + ((ear[i] / 511.0 * zfactor - xth[jl]) * 
                                     (spt[jl + 1] - spt[jl]) / 
                                     (xth[jl + 1] - xth[jl])                 )
            else:
                prim[i] = spt[0]
    for i in range(1, ne):
        photar[i] = (0.5 * (prim[i] / ear[i]**2 + prim[i - 1] / ear[i - 1]**2) 
                         * (ear[i] - ear[i - 1]) * normfac                    )

    return photar



def _thcompton(tempbb, theta, gamma):
    """
    This function was adapted by ADT from the subroutine thcompton in
    donthcomp.f, distributed with XSpec.
    Nthcomp documentation:
    https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSmodelNthcomp.html
    Refs:
    Zdziarski, Johnson & Magdziarz 1996, MNRAS, 283, 193,
    as extended by Zycki, Done & Smith 1999, MNRAS 309, 561

    The original fortran documentation for this subroutine is included below:

    Thermal Comptonization; solves Kompaneets eq. with some
    relativistic corrections. See Lightman \ Zdziarski (1987), ApJ
    The seed spectrum is a blackbody.
    version: January 96

    #c  input parameters:
    #real * 8 tempbb,theta,gamma
    """
    #c use internally Thomson optical depth
    tautom = np.sqrt(2.250 + 3.0 / (theta * ((gamma + .50)**2 - 2.250))) - 1.50

    # Initialise arrays
    dphdot = np.zeros(900); rel = np.zeros(900); c2 = np.zeros(900)
    sptot  = np.zeros(900); bet = np.zeros(900); x  = np.zeros(900)

    #c JMAX  -  # OF PHOTON ENERGIES
    #c delta is the 10 - log interval of the photon array.
    delta = 0.02
    deltal = delta * np.log(10.0)
    xmin = 1e-4 * tempbb
    xmax = 40.0 * theta
    jmax = min(899, int(np.log10(xmax / xmin) / delta) + 1)

    #c X  -  ARRAY FOR PHOTON ENERGIES
    # Energy array is normalized by 511 keV, the rest energy of an electron
    x[:(jmax + 1)] = xmin * 10.0**(np.arange(jmax + 1) * delta)

    #c compute c2(x), and rel(x) arrays
    #c c2(x) is the relativistic correction to Kompaneets equation
    #c rel(x) is the Klein - Nishina cross section divided by the
    #c Thomson crossection
    for j in range(0, jmax):
        w = x[j]
    #c c2 is the Cooper's coefficient calculated at w1
    #c w1 is x(j + 1 / 2) (x(i) defined up to jmax + 1)
        w1 = np.sqrt(x[j] * x[j + 1])
        c2[j] = (w1**4 / (1.0 + 4.60 * w1 + 1.1 * w1 * w1))
        if (w <= 0.05):
            #c use asymptotic limit for rel(x) for x less than 0.05
            rel[j] = (1.0 - 2.0 * w + 26.0 * w * w * 0.2)
        else:
            z1 = (1.0 + w) / w**3
            z2 = 1.0 + 2.0 * w
            z3 = np.log(z2)
            z4 = 2.0 * w * (1.0 + w) / z2
            z5 = z3 / 2.0 / w
            z6 = (1.0 + 3.0 * w) / z2 / z2
            rel[j] = (0.75 * (z1 * (z4 - z3) + z5 - z6))

    #c the thermal emission spectrum
    jmaxth = min(900, int(np.log10(50 * tempbb / xmin) / delta))
    if (jmaxth > jmax):
       jmaxth = jmax
    planck = 15.0 / (np.pi * tempbb)**4
    dphdot[:jmaxth] = planck * x[:jmaxth]**2 / (np.exp(x[:jmaxth] / tempbb)-1)

    #c compute beta array, the probability of escape per Thomson time.
    #c bet evaluated for spherical geometry and nearly uniform sources.
    #c Between x = 0.1 and 1.0, a function flz modifies beta to allow
    #c the increasingly large energy change per scattering to gradually
    #c eliminate spatial diffusion
    jnr  = int(np.log10(0.10 / xmin) / delta + 1)
    jnr  = min(jnr, jmax - 1)
    jrel = int(np.log10(1 / xmin) / delta + 1)
    jrel = min(jrel, jmax)
    xnr  = x[jnr - 1]
    xr   = x[jrel - 1]
    for j in range(0, jnr - 1):
        taukn = tautom * rel[j]
        bet[j] = 1.0 / tautom / (1.0 + taukn / 3.0)
    for j in range(jnr - 1, jrel):
        taukn = tautom * rel[j]
        arg = (x[j] - xnr) / (xr - xnr)
        flz = 1 - arg
        bet[j] = 1.0 / tautom / (1.0 + taukn / 3.0 * flz)
    for j in range(jrel, jmax):
        bet[j] = 1.0 / tautom

    dphesc = _thermlc(tautom, theta, deltal, x, jmax, dphdot, bet, c2)

    #c     the spectrum in E F_E
    for j in range(0, jmax - 1):
        sptot[j] = dphesc[j] * x[j]**2

    return x, jmax, sptot



def _thermlc(tautom, theta, deltal, x, jmax, dphdot, bet, c2):
    """
    This function was adapted by ADT from the subroutine thermlc in 
    donthcomp.f, distributed with XSpec.
    Nthcomp documentation:
    https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSmodelNthcomp.html
    Refs:
    Zdziarski, Johnson & Magdziarz 1996, MNRAS, 283, 193,
    as extended by Zycki, Done & Smith 1999, MNRAS 309, 561

    The original fortran documentation for this subroutine is included below:

    This program computes the effects of Comptonization by
    nonrelativistic thermal electrons in a sphere including escape, and
    relativistic corrections up to photon energies of 1 MeV.
    the dimensionless photon energy is x = hv / (m * c * c)

    The input parameters and functions are:
    dphdot(x), the photon production rate
    tautom, the Thomson scattering depth
    theta, the temperature in units of m*c*c
    c2(x), and bet(x), the coefficients in the K - equation and the
      probability of photon escape per Thomson time, respectively,
      including Klein - Nishina corrections
    The output parameters and functions are:
    dphesc(x), the escaping photon density
    """
    dphesc = np.zeros(900)  # Initialise the output
    a = np.zeros(900); b   = np.zeros(900); c = np.zeros(900)
    d = np.zeros(900); alp = np.zeros(900); u = np.zeros(900)
    g = np.zeros(900); gam = np.zeros(900)

    #c u(x) is the dimensionless photon occupation number
    c20 = tautom / deltal

    #c determine u
    #c define coefficients going into equation
    #c a(j) * u(j + 1) + b(j) * u(j) + c(j) * u(j - 1) = d(j)
    for j in range(1, jmax - 1):
        w1 = np.sqrt( x[j] * x[j + 1] )
        w2 = np.sqrt( x[j - 1] * x[j] )
        #c  w1 is x(j + 1 / 2)
        #c  w2 is x(j - 1 / 2)
        a[j] =  -c20 * c2[j] * (theta / deltal / w1 + 0.5)
        t1 =  -c20 * c2[j] * (0.5 - theta / deltal / w1)
        t2 = c20 * c2[j - 1] * (theta / deltal / w2 + 0.5)
        t3 = x[j]**3 * (tautom * bet[j])
        b[j] = t1 + t2 + t3
        c[j] = c20 * c2[j - 1] * (0.5 - theta / deltal / w2)
        d[j] = x[j] * dphdot[j]

    #c define constants going into boundary terms
    #c u(1) = aa * u(2) (zero flux at lowest energy)
    #c u(jx2) given from region 2 above
    x32 = np.sqrt(x[0] * x[1])
    aa = (theta / deltal / x32 + 0.5) / (theta / deltal / x32 - 0.5)

    #c zero flux at the highest energy
    u[jmax - 1] = 0.0

    #c invert tridiagonal matrix
    alp[1] = b[1] + c[1] * aa
    gam[1] = a[1] / alp[1]
    for j in range(2, jmax - 1):
        alp[j] = b[j] - c[j] * gam[j - 1]
        gam[j] = a[j] / alp[j]
    g[1] = d[1] / alp[1]
    for j in range(2, jmax - 2):
        g[j] = (d[j] - c[j] * g[j - 1]) / alp[j]
    g[jmax - 2] = (d[jmax - 2] - a[jmax - 2] * u[jmax - 1] 
                               - c[jmax - 2] * g[jmax - 3]) / alp[jmax - 2]
    u[jmax - 2] = g[jmax - 2]
    for j in range(2, jmax + 1):
        jj = jmax - j
        u[jj] = g[jj] - gam[jj] * u[jj + 1]
    u[0] = aa * u[1]
    #c compute new value of dph(x) and new value of dphesc(x)
    dphesc[:jmax] = x[:jmax] * x[:jmax] * u[:jmax] * bet[:jmax] * tautom

    return dphesc



def non_thermal(E_peak, Gamma):
    """
    Return a model spectrum of the non-thermal component, i.e. the inverse-
    Compton scattered X-ray power-law tail.  This function calls the
    "_donthcomp" function, which calls "_thcompton", which calls "_thermlc".
      E_peak: The energy of the peak of the BBB disk emission, in log10(E/keV).
              The peak is in a plot of log10(keV*((keV/cm2/s)/keV)) vs
              log10(E/keV), i.e. log10(E*F_E) vs log10(bin energy in keV).
      Gamma:  The negative of the power-law slope in a plot of 
              log10(phtns/cm2/s/keV) vs. log10(bin energy in keV) for the
              non-thermal component.
    Returns three vectors:
    - a vector of bin energies in keV.  Energies are the bin centre when
      considered in log(E/keV) space.
    - a vector of bin widths in keV, which are constant in log(E/keV) space.
    - a vector containing the total flux (keV/cm2/s) in each energy bin.
    The returned flux vector is normalized such that summing the output fluxes
    in the range 0.01 < E (keV) < 20 will give 1.  If there is no flux in the
    range 0.01 < E (keV) < 20, a zero flux vector is returned.
    """
 
    # Determine energy bins, ensuring that the range covers the vast majority
    # of the flux for both the BBB and the power-law component:
    E_vec, bin_widths = _find_E_vec(E_peak)
    # Bin energies (bin centres in log space) and bin widths are in keV
    ear = E_vec + (bin_widths / 2.0)  # Vector of bin energies at right ("ear")
    
    # We need to estimate the temperature of the disk at the coronal radius t0,
    # which is the seed blackbody temperature used in optxagnf in the
    # high-energy Compton calculation.  The following reasonable linear
    # fit is based on outputting t0 from optxagnf for a range of models:
    lg10_T_cor_keV = 0.956432 * E_peak - 0.387753  # E_peak in log10(E/keV)
    # where lg10_T_cor_keV is the (colour-corrected) disk temperature at r_cor
    T_cor = 10**lg10_T_cor_keV  # keV

    # For testing: from optxagnf (L, M, r_cor) = (0.4 LEdd, 10^7 MSun, 10 r_g),
    # (and zero spin), t0 = T_cor = 451255.4 K including colour correction

    # Make parameter list to input into the _donthcomp function:
    param_list = [ 0,      # 0: Dummy item - for fortran-like indexing from 1
                   Gamma,  # 1: photon spectral index
                   100,    # 2: plasma temperature in keV (assumption)
                   T_cor,  # 3: blackbody temperature in keV
                   0,      # 4: type of seed spectrum (0 - blackbody)
                   0    ]  # 5: redshift
    NT_photar = _donthcomp(ear, param_list)   # photons/cm2/s

    # Convert photons/cm2/s -> keV/cm2/s in each bin
    F1 = NT_photar * E_vec  # Energy flux in keV/cm2/s
    # Note that we assume that all photons in a bin have the same energy.
    # Normalization, such that the total for 0.01 < E (keV) < 20 is 1: 
    norm_range = (E_vec > E_norm_range[0]) & (E_vec < E_norm_range[1])
    E_norm_range_sum = np.sum( F1[norm_range] )  # keV/cm2/s
    if E_norm_range_sum > 0:  # If there is flux in the relevant energy range:
        F1 = F1 / E_norm_range_sum
    else: # If there is no flux in the relevant energy range:
        F1 *= 0 # Return a zero vector, since we can't normalize the spectrum

    # We want to output only the following energy range:
    o_indices = (E_vec > output_range[0]) & (E_vec < output_range[1])  

    # Summing the list of output fluxes (keV/cm2/s for each bin) should
    # give approximately 1 keV/cm2/s.  Because the normalization was over a
    # subset of the full range, the sum of output fluxes should be somewhat
    # more than 1.  If the sum is zero, then there is no flux in the relevant 
    # ionising range, and the whole flux vector was set to zero.

    # Return bin energies (keV), bin widths (keV), total bin fluxes (keV/cm2/s)
    return E_vec[ o_indices ], bin_widths[ o_indices ], F1[ o_indices ]



def disk(E_peak):
    """
    Return the oxaf model spectrum of the Big Blue Bump (BBB) disk emission.
        E_peak: The energy of the peak of the BBB, in log10(E/keV).
                The peak is in a plot of log10(keV*((keV/cm2/s)/keV)) vs
                log10(E/keV), i.e. log10(E*F_E) vs log10(bin energy in keV).
    Returns three vectors:
    - a vector of bin energies in keV.  Energies are the bin centre when
      considered in log(E/keV) space.
    - a vector of bin widths in keV, which are constant in log(E/keV) space.
    - a vector containing the total energy flux (keV/cm2/s) in each energy bin.
    The returned flux vector is normalized so that summing the output fluxes
    in the range 0.01 < E (keV) < 20 will give 1.  If there is no flux in the
    range 0.01 < E (keV) < 20, a zero flux vector is returned.
    """
    # This working is based on the spreadsheet BBB_modelling_v0.1.xls
    
    # Determine energy bins, ensuring that the range covers the vast majority
    # of the flux for both the BBB and the power-law component:
    E_vec, bin_widths = _find_E_vec(E_peak)
    # Bin energies (bin centres in log space) and bin widths are in keV
    lg_E = np.log10(E_vec)

    # The following is a matrix of coefficients for 6th-order polynomial fits
    # to the BBB spectra.  Six different spectra were fit, for six values of
    # E_peak.  Each row of the matrix contains the seven polynomial
    # coeffiecients for one of the six spectra.
    B_params = np.array( [
    #   B_x6        B_x5        B_x4        B_x3         B_x2       B_x1       B_x0
    [-1.2372E-1, -6.0806E-1, -1.3943E+0, -1.9966E+0, -1.9697E+0, +7.7015E-3, +5.1353E-5],
    [-8.8187E-2, -5.0491E-1, -1.4647E+0, -2.2006E+0, -2.0531E+0, +3.1928E-2, +1.5599E-5],
    [-1.2360E-1, -5.6311E-1, -9.0134E-1, -7.1795E-1, -8.5483E-1, +1.1660E-1, +8.3705E-3],
    [-7.9301E-2, -5.2860E-1, -1.3232E+0, -1.4923E+0, -1.1230E+0, -3.0312E-2, +4.9524E-3],
    [-7.8550E-2, -6.0659E-1, -1.7534E+0, -2.2425E+0, -1.4240E+0, +4.0136E-2, -3.3227E-3],
    [-2.5224E-2, -3.1527E-1, -1.4363E+0, -2.7712E+0, -2.1937E+0, +3.1206E-1, -6.3991E-4] ]) 
    
    # The following E_peak values (log10(E/keV)) correspond to the BBB fitted
    # for each row of the above matrix:
    fitted_peaks = np.array( [-3.025, -2.403, -1.878, -1.493, -1.177, -0.722] ) 
    n = fitted_peaks.size
    
    # We interpolate the spectral shape between the six fitted spectra.
    # "Above": Index of nearest fitted peak above the input E_peak value
    # "Below": Index of nearest fitted peak below the input E_peak value
    # "weight_A": Weight to give the fitted spectrum corresponding to the
    #             nearest fitted peak above the input E_peak value 
    # "weight_B": Weight to give the fitted spectrum corresponding to the
    #             nearest fitted peak below the input E_peak value     
    if E_peak < fitted_peaks[0]:
        Above = 0; Below = 0
        weight_A = 0.5; weight_B = 0.5;
    elif E_peak > fitted_peaks[n-1]:
        Above = n-1; Below = n-1
        weight_A = 0.5; weight_B = 0.5;
    elif E_peak in fitted_peaks:
        Above = (np.abs(fitted_peaks - E_peak)).argmin()
        Below = Above
        weight_A = 0.5; weight_B = 0.5;
    else:
        for i in np.arange(n-1):
            if fitted_peaks[i] < E_peak < fitted_peaks[i+1]:
                Below = i; Above = i+1
                dist_A = abs(fitted_peaks[Above] - E_peak);
                dist_B = abs(fitted_peaks[Below] - E_peak);
                # Weights according to "closeness" (weights are normalized):
                # large dist_B (small dist_A) -> larger weight for "A" spectrum
                weight_A = dist_B / (dist_A + dist_B)
                weight_B = dist_A / (dist_A + dist_B)
                break

    # Polynomial coeffiecients for modeling BBB:
    AB_x6, AB_x5, AB_x4, AB_x3, AB_x2, AB_x1, AB_x0 = B_params[Above,:].tolist()
    BB_x6, BB_x5, BB_x4, BB_x3, BB_x2, BB_x1, BB_x0 = B_params[Below,:].tolist()
    
    # Calculate un-normalized E*F_E for each energy bin,
    # in log10(keV*((keV/cm2/s)/keV)):
    F1 = ( weight_A * 
           ( AB_x6*(lg_E-E_peak)**6 + AB_x5*(lg_E-E_peak)**5 +
             AB_x4*(lg_E-E_peak)**4 + AB_x3*(lg_E-E_peak)**3 + 
             AB_x2*(lg_E-E_peak)**2 + AB_x1*(lg_E-E_peak)    + AB_x0 ) + 
           weight_B  * 
           ( BB_x6*(lg_E-E_peak)**6 + BB_x5*(lg_E-E_peak)**5 + 
             BB_x4*(lg_E-E_peak)**4 + BB_x3*(lg_E-E_peak)**3 +
             BB_x2*(lg_E-E_peak)**2 + BB_x1*(lg_E-E_peak)    + BB_x0 )    )

    F1 = 10**F1  # E*F_E for each bin (keV*((keV/cm2/s)/keV))
    F1 = F1 / E_vec * bin_widths  # Total flux, F in keV/cm2/s for each bin
    # Normalization, such that the total for 0.01 < E (keV) < 20 is 1: 
    norm_range = (E_vec > E_norm_range[0]) & (E_vec < E_norm_range[1])
    E_norm_range_sum = np.sum( F1[norm_range] )  # keV/cm2/s
    if E_norm_range_sum > 0:  # If there is flux in the relevant energy range:
        F1 = F1 / E_norm_range_sum
    else: # If there is no flux in the relevant energy range:
        F1 *= 0 # Return a zero vector, since we can't normalize the spectrum
    
    # We output only the following energy range:
    o_indices = (E_vec > output_range[0]) & (E_vec < output_range[1])   

    # Summing the list of output fluxes (keV/cm2/s for each bin) should
    # give approximately 1 keV/cm2/s.  Because the normalization was over a
    # subset of the full range, the sum of output fluxes should be somewhat
    # more than 1.  If the sum is zero, then there is no flux in the relevant 
    # ionising range, and the whole flux vector was set to zero.

    # Return bin energies (keV), bin widths (keV), total bin fluxes (keV/cm2/s)
    return E_vec[ o_indices ], bin_widths[ o_indices ], F1[ o_indices ]



def full_spectrum(E_peak, Gamma, p_NT):
    """
    Return a spectrum including both the Big Blue Bump (BBB) disk component and
    the non-thermal component.  This function implements the full oxaf model,
    i.e. returns a spectrum given only three non-degenerate parameters.
        E_peak: The energy of the peak of the BBB, in log10(E/keV).
                The peak is in a plot of log10(keV*((keV/cm2/s)/keV)) vs
                log10(E/keV), i.e. log10(E*F_E) vs log10(bin energy in keV).
        Gamma:  The negative of the power-law slope in a plot of 
                log10(phtns/cm2/s/keV) vs. log10(bin energy in keV) for the
                non-thermal component.
        p_NT:   The proportion of the total flux over the range
                0.01 < E (keV) < 20 which is in the non-thermal component, with
                (1 - p_NT) being the proportion in the BBB disk component.
    Returns three vectors:
    - a vector of bin energies in keV.  Energies are the bin centre when
      considered in log(E/keV) space.
    - a vector of bin widths in keV, which are constant in log(E/keV) space.
    - a vector containing the total flux (keV/cm2/s) in each energy bin.
    The returned flux vector is normalized so that summing the output fluxes
    in the range 0.01 < E (keV) < 20 gives 1.
    """
    
    # Check the input parameters to ensure they're not outrageously wrong:
    if not (-5.0 <= E_peak <= 2.0):
        raise ValueError("E_peak must be between -5 and 2")
    if not (1.0 < Gamma <= 4.0):
        raise ValueError("Gamma must be between 1 and 4")
    if not (0.0 <= p_NT <= 1.0):
        raise ValueError("p_NT must be between 0 and 1")

    E_vec, B_vec, F_BBB = disk(E_peak) # keV, keV, keV/cm2/s
    E_vec, B_vec, F_NT  = non_thermal(E_peak, Gamma) # keV, keV, keV/cm2/s
    # Now F_BBB and F_NT are normalized vectors of total fluxes (keV/cm2/s) in
    # each bin.  Both are normalized so that the total flux in each is 1 over
    # the range 0.01 < E (keV) < 20.

    return E_vec, B_vec, (1.0-p_NT)*F_BBB + p_NT*F_NT



def _main():
    """
    Execute this function if the module is being run as a command line script.
    See the help string below for documentation of this feature.
    """
    import sys
    
    args = sys.argv # Obtain the command-line arguments
    if len(args) != 4:  # The zeroth argument is the name of the script
        print("""Usage:  python oxaf.py E_peak Gamma p_NT
        
        Returns the oxaf model spectrum of AGN continuum emission, including
        both the Big Blue Bump (BBB) disk emission and the non-thermal emission.
        E_peak: The energy of the peak of the BBB, in log10(E/keV).
                The peak is in a plot of log10(keV*((keV/cm2/s)/keV)) vs
                log10(E/keV), i.e. log10(E*F_E) vs log10(bin energy in keV).
        Gamma:  The negative of the power-law slope in a plot of 
                log10(phtns/cm2/s/keV) vs. log10(bin energy in keV) for the
                non-thermal component.
        p_NT:   The proportion of the total flux over the range
                0.01 < E (keV) < 20 which is in the non-thermal component, with
                (1 - p_NT) being the proportion in the BBB disk component.
        
        Prints three columns of data to stdout:
        - The bin energies in keV.  Energies are the bin centre when
          considered in log10(E/keV) space.
        - The bin widths in keV, which are constant in log(E/keV) space.
        - The total energy flux (keV/cm2/s) in each energy bin.
        The returned fluxes are normalized so that summing the list of  
        output fluxes over the range 0.01 < E (keV) < 20 gives 1.
        """)
        return

    E_peak = np.float(args[1])
    Gamma  = np.float(args[2])
    p_NT   = np.float(args[3])

    # Generate the model spectrum:
    E, B, F = full_spectrum(E_peak, Gamma, p_NT)
    # Print the result columns to stdout:
    print("Energy       Bin_width    Normalized_flux")
    print("keV          keV          keV/cm2/s")
    for E_i, B_i, F_i in zip(E, B, F):
        print("{0:.06e} {1:.06e} {2:.06e}".format(E_i, B_i, F_i))



if __name__ == "__main__":
    # If we're running this module as a script, call the "_main" function:
    _main()
    
