# Created by Yen-Hsun Lin (Academia Sinica) in 03/2024.
# Copyright (c) 2024 Yen-Hsun Lin.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version (see <http://www.gnu.org/licenses/>).
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.


import numpy as _np
import vegas as _vegas
#from .sysmsg import FlagError
from snorer.sysmsg import FlagError
from .constant import constant
from .galDensity import galacticAreaDensityFit,galacticAreaDensity,cosmicAgeFit
from .galMassFunction import dnG,_E,rhoDotSFR



##########################################################################
#                                                                        #
#   General Classes and Functions for Numerics                           #
#                                                                        #
##########################################################################


# ----- Dark matter halo profile -----

class haloSpike(constant):  
    
    def __init__(self):
        """
        Class for DM halo with spike scaled to arbitrary galactic mass MG
        """
        pass
    
    def _M_sigma(self,mBH):
        """
        Faber–Jackson law for black holes

        In
        ---
        mBH: SMBH mass, Msun

        Out
        ---
        sigma: stellar velocity dispersion, km/s
        """
        return 200*(mBH/1e8/1.9)**(1/5.1)
    
    def _rh(self,MG,eta):
        """
        SMBH influence radius

        In
        ---
        MG: galactic mass, Msun
        eta: the ratio of MG/Mhalo

        Out
        ---
        rh: influence radisu, kpc
        """
        mBH = massBH(MG,eta)
        sigma = self._M_sigma(mBH)
        rh = self.G*mBH/sigma**2 # pc
        return rh*1e-3 # kpc

    def _normN(self,mBH,rh) -> float:
        """
        The normalization N
        
        In
        ------
        mBH: SMBH mass, Msun
        rh: SMBH influence radius, kpc
        
        Out
        ------
        normalization N
        """
        Rs = radiusSchwarzschild(mBH)
        ri = 4*Rs
        alpha = 3/2
        
        def _fa(r,alpha):
            return (r**3/(3 - alpha) + 12*Rs*r**2/(alpha - 2) - 48*Rs**2*r/(alpha - 1) + 64*Rs**3/alpha)/r**alpha

        fh,fi = _fa(rh,alpha),_fa(ri,alpha)
        norm = mBH*self.Msun/4/_np.pi/(fh - fi)
        return norm

    def _radiusSpike(self,mBH,rh,rhos,rs) -> float:
        """
        Get the spike radius

        In
        ------
        mBH: SMBH mass, Msun
        rh: SMBH influence radius, kpc
        rhos: characteristic density, MeV/cm^3
        rs: characteristic radius, kpc

        Out
        ------
        R_spike: Spike radius, kpc
        """
        N = self._normN(mBH,rh)
        rhos = rhos*self.kpc2cm**3
        return (N/rhos/rs)**(3/4)*rh**(5/8)
    
    def _rhoPrime(self,r,mBH,rh,rhos,rs) -> float:
        Rs = radiusSchwarzschild(mBH)
        ri = 4*Rs
        N = self._normN(mBH,rh)
        Rsp = self._radiusSpike(mBH,rh,rhos,rs)
        rhoN = N/rh**(3/2)
        rhoNp = rhoN*(rh/Rsp)**(7/3)
        if ri <= r < rh:
            rhoP = rhoN*(1 - ri/r)**3*(rh/r)**(3/2)
        else:
            rhoP = rhoNp*(Rsp/r)**(7/3)
        return rhoP/self.kpc2cm**3

    def _nxSpike(self,r,mx,MG,sigv,tBH,rhosMW,rsMW,eta,rh) -> float:
        """
        DM number density with spike in the center

        In
        ------
        r: distance to GC, kpc
        mx: DM mass, MeV
        MG: The galactic stellar mass, Msun
        sigv: DM annihilation cross section, in the unit of 1e-26 cm^3/s
            None indicates no annihilation
        tBH: SMBH age, default 1e9 years
        rhosMW: The MW characteristic density, MeV/cm^3
        rsMW: The MW characteristic radius, kpc
        eta: the ratio of MG/Mhalo
        rh: SMBH influence radius, kpc

        Out
        ------
        density: #/cm^3
        """
        #mHalo = eta*MG
        mBH = massBH(MG,eta)
        Rs = radiusSchwarzschild(mBH)
        ri = 4*Rs
        rs = get_rs(MG,rsMW)  # get the scaled rs from MG
        
        if sigv is None:
            Rsp = self._radiusSpike(mBH,rh,rhosMW,rs)
            if r < ri:
                return 0
            elif ri <= r < Rsp:
                return self._rhoPrime(r,mBH,rh,rhosMW,rs)/mx
            else:
                return rhox(r,rhosMW,rs)/mx
        else:
            Rsp = self._radiusSpike(mBH,rh,rhosMW,rs)
            sigv = sigv*1e-26
            rhoc = mx/sigv/tBH/self.year2Seconds
            if r < ri:
                return 0
            elif ri <= r < Rsp:
                rhoP = self._rhoPrime(r,mBH,rh,rhosMW,rs)
                return rhoP*rhoc/(rhoP + rhoc)/mx
            else:
                rhoDM = rhox(r,rhosMW,rs)
                return rhoDM*rhoc/(rhoDM + rhoc)/mx
    
    def __call__(self,r,mx,MG,sigv,tBH,rhosMW,rsMW,eta):
        rh = self._rh(MG,eta) # SMBH influence radius in kpc
        return self._nxSpike(r,mx,MG,sigv,tBH,rhosMW,rsMW,eta,rh)


def rhox(r,rhos,rs) -> float:
    """
    NFW DM density at given r
    
    In
    ------
    r: distance to GC, kpc
    rhos: The characteristic density, MeV/cm^3
    rs: The characteristic radius, kpc
    
    Out
    ------
    rhox: DM density at r, MeV/cm^3
    """
    rr = r/rs
    return rhos/(rr*(1 + rr)**2)


def get_rmax(MG) -> float:
    """
    Obtain the halo radius for arbitrary MG, scaled from MW
    
    In
    ------
    MG: The galactic stellar mass, Msun
    
    Out
    ------
    rmax: kpc
    """
    if MG is None:
        return constant.Rhalo
    else:
        return (MG/constant.Mmw)**(1/3)*constant.Rhalo


def get_rs(MG,rsMW=24.42) -> float:
    """
    Obtain the characteristic radius for arbitrary MG
    
    In
    ------
    MG: Galactic stellar mass, Msun
    rsMW: The MW characteristic radius, kpc
    
    Out
    ------
    rs: kpc
    """
    if MG is None:
        return rsMW
    else:
        return (MG/constant.Mmw)**(1/3)*rsMW


def nxNFW(r,mx,rhosMW=184,rsMW=24.42,MG=None) -> float:
    """
    DM number density at r for arbitrary MG without spike
    
    In
    ------
    r: distance to GC, kpc
    mx: DM mass, MeV
    rhosMW: The MW characteristic density, MeV/cm^3
    rsMW: The MW characteristic radius, kpc
    MG: Galactic stellar mass, Msun
        Default is None, and implies MW case
    eta: the ratio of MG/Mhalo
        Default is 24.38 which corresponds to MW case
    
    Out
    ------
    number density: per cm^3
    """
    if MG is None:
        return rhox(r,rhosMW,rsMW)/mx
    else:
        rs = get_rs(MG,rsMW)
        return rhox(r,rhosMW,rs)/mx


def massBH(MG,eta=24.38) -> float:
    """
    Estimate of SMBH mass from MG

    In
    ------
    MG: Galactic stellar mass, Msun
    eta: the ratio of MG/Mhalo
        Default is 24.38 which corresponds to MW case

    Out
    ------
    SMBH mass: Msun
    """
    if MG is None:
        # MW case
        return 4.3e6
    else:
        return 7e7*(eta*MG/1e12)**(4/3)


def radiusSchwarzschild(mBH) -> float:
    """
    Calculating the Schawarzschild radius 

    In
    ------
    mBH: Black hole mass, Msun

    Ou
    ------
    Rs: Schwarzschild radius, kpc
    """
    mBH = mBH*constant.Msun_kg
    Rs = mBH*1.48e-25/constant.kpc2cm  # convert Rs into kpc
    return Rs


def dmNumberDensity(r,mx,MG,is_spike=True,sigv=None,tBH=1e10,rhosMW=184,rsMW=24.42,eta=24.3856) -> float:
    """
    Obtain the DM number density at given r with arbitrary MG
    
    In
    ------
    r: distance to GC, kpc
    mx: DM mass, MeV
    MG: The galactic stellar mass, Msun
    is_spike: Turn on/off spike feature, bool
    sigv: DM annihilation cross section, in the unit of 1e-26 cm^3/s
        None indicates no annihilation
    tBH: SMBH age, years
    rhosMW: The MW characteristic density, MeV/cm^3
    rsMW: The MW characteristic radius, kpc
    eta: the ratio of MG/Mhalo
    
    Out
    ------
    number density: 1/cm^3
    """
    if is_spike is True:
        nx = haloSpike()
        return nx(r,mx,MG,sigv,tBH,rhosMW,rsMW,eta)
    elif is_spike is False:       
        return nxNFW(r,mx,rhosMW,rsMW,MG)
    else:
        raise FlagError('Flag \'is_spike\' must be a boolean.')



# ----- Supernova neutrinos and propagation geometry -----

def _get_r(l,R,theta) -> float:
    """
    Get the distance r between boosted point and GC
    
    In
    ------
    l: SN neutrino propagation length, kpc
    R: Distance between SN and GC, kpc
    theta: polar angle, rad
    
    Out
    ------
    r: kpc
    """
    return _np.sqrt(l**2 + R**2 - 2*l*R*_np.cos(theta))


def snNuEenergy(Tx,mx,thetaCM) -> float:
    """
    Get the required incoming SN neutrino energy Ev
    
    In
    ------
    Tx: DM kinetic energy, MeV
    mx: DM mass, MeV
    thetaCM: Scattering angle in CM frame, rad
    
    Out
    ------
    Ev: MeV
    """
    c2 = _np.cos(thetaCM/2)**2
    return Tx*(1 + _np.sqrt(1 + 2*c2*mx/Tx))/2/c2


def _dEv(Tx,mx,thetaCM) -> float:
    """
    Get the slope of Ev versus Tx, dEv/dTx
    
    In
    ------
    Tx: DM kinetic energy, MeV
    mx: DM mass, MeV
    thetaCM: Scattering angle in CM frame, rad
    
    Out
    ------
    dEv/dTx: dimensionless
    """
    c2 = _np.cos(thetaCM/2)**2
    x = mx/Tx
    return (1 + (1 + c2*x)/_np.sqrt(2*c2*x + 1))/2/c2


def vBDM(Tx,mx) -> float:
    """
    Get the BDM velocity in the unit of light speed
    
    In
    ------
    Tx: DM kinetic energy, MeV
    mx: DM mass, MeV
    
    Out
    ------
    velocity: in the unit of c
    """
    return _np.sqrt(Tx*(Tx + 2*mx))/(Tx + mx)


#def dsigma0(Ev,thetaCM) -> float:
#    """
#    Differential DM-neutrino scattering cross section in CM frame
#    
#    In
#    ------
#    Ev: incoming SN neutrino energy, MeV
#    thetaCM: Scattering angle in CM frame, rad
#    
#    Out
#    ------
#    dsigma0: cm^2 per steradian
#    """
#    # this is energy-independent cross section
#    # divided by 4*np implying isotropic in CM frame
#    return print('Academics is a place full of PUA!')
#    return 1e-35


def supernovaNuFlux(Ev,l,is_density=False) -> float:
    """
    SN neutrino flux after propagating a distance l
    
    Input
    ------
    Ev: SN neutrino energy, MeV
    l: propagation distance, kpc
    is_density: output neutrino number denisty within shell
    
    Output
    ------
    flux: #/Ev/cm^2/s, if is_density is False
    number density: #/Ev/cm^3, if is_density is True
    """
    Lv = constant.Lv*constant.erg2MeV
    l = l*constant.kpc2cm
    
    #Fermi-Dirac distribution
    def _fv(Ev,Tv):
        exponent = Ev/Tv - 3
        # setup a cutoff value when the exponent beyon the validatiy of float64
        if exponent <= 709.782:
            return (1/18.9686)*Tv**(-3)*(Ev**2/(_np.exp(exponent) + 1))
        else:
            return 0
    
    # distributions for nu_e and anti-nu_e
    nue_dist = _fv(Ev,2.76)/11
    nueb_dist = _fv(Ev,4.01)/16
    # distributions for the rest 4 species
    nux_dist = _fv(Ev,6.26)/25
    
    L = Lv/(4*_np.pi*l**2)
    flux = L*(nue_dist + nueb_dist + 4*nux_dist)
    if is_density is False:
        return flux
    elif is_density is True:
        return flux/constant.c
    else:
        raise FlagError('Flag \'is_density\' must be a boolean.')



# ----- Diffuse boosted dark matter -----

class dbdmSpectrum(constant):
    
    def __init__(self):
        pass
    
    def _diffSpectrum(self,Tx,mx,MG,R,l,theta,thetaCM,is_spike,sigv,tBH,rhosMW,rsMW,eta):
        """
        dNx/dTx
        """
        r = _get_r(l,R,theta)
        if 1e-10 <= r < 100:
            Ev = snNuEenergy(Tx,mx,thetaCM)
            dEvdTx = _dEv(Tx,mx,thetaCM)
            vx = vBDM(Tx,mx)  #  
            nx = dmNumberDensity(r,mx,MG,is_spike,sigv,tBH,rhosMW,rsMW,eta)
            dsigma0 = constant.sigma0  # differential DM-nu cross section in CM frame, cm^2/sr
            return l**2*_np.sin(theta)*_np.sin(thetaCM)*nx*dsigma0*supernovaNuFlux(Ev,l)*(dEvdTx*vx)
        else:
            return 0
    
    def _dbdmSpectrum(self,z,MG,Tx,mx,R,l,theta,thetaCM,is_spike,sigv,rhosMW,rsMW,eta) -> float:
        """
        DBDM spectrume yielded by SN at arbitrary position R
        """
        Txp = (1 + z)*Tx 
        tBH = cosmicAgeFit(z)*1e9 # convert to years
        if Txp < 150:  # discard the BDM signature if it requires Ev > 150 MeV at z 
            m = _np.log10(MG)
            return dnG(m,z)*rhoDotSFR(z)*self._diffSpectrum(Txp,mx,
                                                            MG,R,l,theta,
                                                            thetaCM,
                                                            is_spike,sigv,tBH,rhosMW,rsMW,eta)/_E(z)
        else:
            return 0
        
    def _dbdmSpectrumWeighted(self,z,MG,Tx,mx,R,l,theta,thetaCM,is_spike,sigv,rhosMW,rsMW,eta,usefit) -> float:
        """
        DBDM spectrume yielded by SN at position R weighted by galactic baryonic distribution
        """
        Txp = (1 + z)*Tx 
        tBH = cosmicAgeFit(z)*1e9 # convert to years
        if Txp < 150:  # discard the BDM signature if it requires Ev > 130 MeV at z
            # adopt fitting data for galactic area density?
            m = _np.log10(MG)
            if usefit is True:
                galArealDensity = galacticAreaDensityFit((R,m))
            elif usefit is False:
                galArealDensity = galacticAreaDensity(R,zRange=[-10,10],MG=MG)
            else:
                raise FlagError('Flag \'usefit\' must be a boolean.')
            return (2*_np.pi*R)*rhoDotSFR(z)*galArealDensity*dnG(m,z)*self._diffSpectrum(Txp,mx,
                                                                                         MG,R,l,theta,
                                                                                         thetaCM,
                                                                                         is_spike,sigv,tBH,rhosMW,rsMW,eta)/_E(z)/MG    
        else:
            return 0
    
    def __call__(self,z,MG,Tx,mx,R,l,theta,thetaCM,is_spike,is_weighted,sigv,rhosMW,rsMW,eta,usefit):
        if is_weighted is True:
            return self._dbdmSpectrumWeighted(z,MG,Tx,mx,R,l,theta,thetaCM,is_spike,sigv,rhosMW,rsMW,eta,usefit)
        elif is_weighted is False:
            return self._dbdmSpectrum(z,MG,Tx,mx,R,l,theta,thetaCM,is_spike,sigv,rhosMW,rsMW,eta)
        else:
            raise FlagError('Flag \'is_weighted\' must be a boolean.')


def flux(Tx,mx,
         R=0,Rmax=30,rmax=30,tau=10,is_spike=True,is_average=True,
         sigv=None,rhosMW=184,rsMW=24.42,eta=24.3856,usefit=True,
         nitn=10,neval=50000):
    """
    DBDM flux for given (Tx,mx) assuming isotropic and energy-independent
    differential DM-nuetrino cross section in CM frame with the value
    1e-35/4/pi cm^2/std
    
    In
    ------
    Tx: BDM kinetic energy, MeV
    mx: DM mass, MeV
    R: Specified SN position on the galactic plane, R=0=GC
        This only works for is_average = False
    Rmax: Maximum radius of galactic plane to be integrated
    rmax: Maximum halo radius to be integrated
    tau: SN duration, s
    is_spike: Including DM spike in the halo, bool
    is_average: SN position weighted by baryonic distribution, bool
    sigv: DM annihilation cross section, in the unit of 1e-26 cm^3/s
        None indicates no annihilation
    rhosMW: NFW characteristic density for MW
    rsMW: NFW characteristic radius for MW
    eta: Mmw/Mhalo for MW
    nitn: Number of chains in vegas
    neval: Number of evaluation in each MCMC chain
    
    Out
    ------
    Flux: per MeV per cm^2 per second
    """
    preFactor = constant.MagicalNumber  # constant.D_H0*0.017/constant.Mmw/rhoDotSFR(0)/1e6/constant.kpc2cm**2/constant.year2Seconds
                                        # 0.017: SN in MW per year; 1e6: converting Mpc^2 to kpc^2
    lmax = Rmax + rmax
    spectrum  = dbdmSpectrum()
    if is_average is True:
        integrator = _vegas.Integrator([[0,8],[1e6,1e12],[0,Rmax],[0,lmax],[0,_np.pi],[0,_np.pi]]) #(z,m,R,l,theta,thetaCM)
        result = integrator(lambda x: spectrum(z=x[0],MG=x[1],Tx=Tx,mx=mx,R=x[2],l=x[3],theta=x[4],thetaCM=x[5],
                                               is_spike=is_spike,is_weighted=is_average,sigv=sigv,
                                               rhosMW=rhosMW,rsMW=rsMW,eta=eta,usefit=usefit),nitn=nitn,neval=neval).mean
        flux = 4*_np.pi**2*tau*result*constant.kpc2cm**3*vBDM(Tx,mx)*preFactor
    elif is_average is False:
        integrator = _vegas.Integrator([[0,8],[1e6,1e12],[0,lmax],[0,_np.pi],[0,_np.pi]]) #(z,m,l,theta,thetaCM)
        result = integrator(lambda x: spectrum(z=x[0],MG=x[1],Tx=Tx,mx=mx,R=R,l=x[2],theta=x[3],thetaCM=x[4],
                                               is_spike=is_spike,is_weighted=is_average,sigv=sigv,
                                               rhosMW=rhosMW,rsMW=rsMW,eta=eta,usefit=usefit),nitn=nitn,neval=neval).mean
        flux = 4*_np.pi**2*tau*result*constant.kpc2cm**3*vBDM(Tx,mx)*preFactor
    else:
        raise FlagError('Flag \'is_average\' must be a boolean.')
    return flux


def event(mx,
          TxRange=[5,30],R=0,Rmax=30,rmax=30,tau=10,is_spike=True,is_average=True,
          sigv=None,rhosMW=184,rsMW=24.42,eta=24.3856,usefit=True,
          nitn=10,neval=50000):
    """
    DBDM event per electron per second for given mx assuming isotropic
    and energy-independent differential DM-nuetrino cross section in CM
    frame with the value 1e-35/4/pi cm^2/std. The total DM-electron cross
    section is assumed 1e-35 cm^2
    
    In
    ------
    mx: DM mass, MeV
    TxRange: [Tx_min,Tx_max], Tx range to be integrated over, MeV
    R: Specified SN position on the galactic plane, R=0=GC
        This only works for is_average = False
    Rmax: Maximum radius of galactic plane to be integrated
    rmax: Maximum halo radius to be integrated
    tau: SN duration, s
    is_spike: Including DM spike in the halo, bool
    is_average: SN position weighted by baryonic distribution, bool
    sigv: DM annihilation cross section, in the unit of 1e-26 cm^3/s
        None indicates no annihilation
    tBH: BH age
    rhosMW: NFW characteristic density for MW
    rsMW: NFW characteristic radius for MW
    eta: Mmw/Mhalo for MW
    nitn: Number of chains in vegas
    neval: Number of evaluation in each MCMC chain
    
    Out
    ------
    Event: per electron per second
    """
    preFactor = constant.MagicalNumber     # constant.D_H0*0.017/constant.Mmw/rhoDotSFR(0)/1e6/constant.kpc2cm**2/constant.year2Seconds
    preFactor *= constant.sigma0*4*_np.pi  # multiplying total DM-electron cross section
    lmax = Rmax + rmax
    spectrum  = dbdmSpectrum()
    if is_average is True:
        integrator = _vegas.Integrator([[0,8],[1e6,1e12],[0,Rmax],[0,lmax],[0,_np.pi],[0,_np.pi],TxRange]) #(z,m,R,l,theta,thetaCM)
        result = integrator(lambda x: spectrum(z=x[0],MG=x[1],Tx=x[6],mx=mx,R=x[2],l=x[3],theta=x[4],thetaCM=x[5],
                                               is_spike=is_spike,is_weighted=is_average,sigv=sigv,
                                               rhosMW=rhosMW,rsMW=rsMW,eta=eta,usefit=usefit)*vBDM(Tx=x[6],mx=mx),
                            nitn=nitn,neval=neval).mean
        event = 4*_np.pi**2*tau*result*constant.kpc2cm**3*preFactor
    elif is_average is False:
        integrator = _vegas.Integrator([[0,8],[1e6,1e12],[0,lmax],[0,_np.pi],[0,_np.pi],TxRange]) #(z,m,l,theta,thetaCM,Tx)
        result = integrator(lambda x: spectrum(z=x[0],MG=x[1],Tx=x[5],mx=mx,R=R,l=x[2],theta=x[3],thetaCM=x[4],
                                               is_spike=is_spike,is_weighted=is_average,sigv=sigv,
                                               rhosMW=rhosMW,rsMW=rsMW,eta=eta,usefit=usefit)*vBDM(Tx=x[5],mx=mx),
                            nitn=nitn,neval=neval).mean
        event = 4*_np.pi**2*tau*result*constant.kpc2cm**3*preFactor
    else:
        raise FlagError('Flag \'is_average\' must be a boolean.')
    return event