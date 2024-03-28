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
from .sysmsg import FlagError
from .constant import constant
from .galDensity import galacticAreaDensityFit,galacticAreaDensity
from .galMassFunction import dnG,_E,rhoDotSFR



##########################################################################
#                                                                        #
#   General Classes and Functions for Numerics                           #
#                                                                        #
##########################################################################


# ----- Dark matter halo profile -----

class haloSpike(constant):  
    """
    Class for DM halo with spike scaled to arbitrary galactic mass MG
    """
    def __init__(self):
        self.rh = 0.65e-3
    
    def _normN(self,mBH) -> float:
        """
        The normalization N
        
        In
        ------
        mBH: SMBH mass, Msun
        
        Out
        ------
        normalization N
        """
        Rs = radiusSchwarzchild(mBH)
        ri = 4*Rs
        rh = 0.65e-3
        alpha = 3/2
        
        def _fa(r,alpha):
            return (r**3/(3 - alpha) + 12*Rs*r**2/(alpha - 2) - 48*Rs**2*r/(alpha - 1) + 64*Rs**3/alpha)/r**alpha

        fh,fi = _fa(rh,alpha),_fa(ri,alpha)
        norm = mBH*self.Msun/4/_np.pi/(fh - fi)
        return norm

    def _radiusSpike(self,mBH,rhos,rs) -> float:
        """
        Get the spike radius

        In
        ------
        mH: SMBH mass, Msun
        rhos: characteristic density, MeV/cm^3
        rs: characteristic radius, kpc

        Out
        ------
        R_spike: Spike radius, kpc
        """
        N = self._normN(mBH)
        rhos = rhos*self.kpc2cm**3
        return (N/rhos/rs)**(3/4)*self.rh**(5/8)
    
    def _rhoPrime(self,r,mBH,rhos,rs) -> float:
        Rs = radiusSchwarzchild(mBH)
        ri = 4*Rs
        N = self._normN(mBH)
        Rsp = self._radiusSpike(mBH,rhos,rs)
        rhoN = N/self.rh**(3/2)
        rhoNp = rhoN*(self.rh/Rsp)**(7/3)
        if ri <= r < self.rh:
            rhoP = rhoN*(1 - ri/r)**3*(self.rh/r)**(3/2)
        else:
            rhoP = rhoNp*(Rsp/r)**(7/3)
        return rhoP/self.kpc2cm**3

    def _nxSpike(self,r,mx,MG,sigv,tBH,rhosMW,rsMW,eta) -> float:
        """
        DM number density with spike in the center

        In
        ------
        r: distance to GC, kpc
        mx: DM mass, MeV
        MG: The galactic stellar mass, Msun
        sigv: DM annihilation cross section, in the unit of cm^-3
            None indicates no annihilation
        tBH: SMBH age, default 1e9 years
        rhosMW: The MW characteristic density, MeV/cm^3
        rsMW: The MW characteristic radius, kpc
        eta: the ratio of MG/Mhalo

        Out
        ------
        density: #/cm^3
        """
        #mHalo = eta*MG
        mBH = massBH(MG,eta)
        Rs = radiusSchwarzchild(mBH)
        ri = 4*Rs
        rs = get_rs(MG,rsMW)  # get the scaled rs from MG
        
        if sigv is None:
            Rsp = self._radiusSpike(mBH,rhosMW,rs)
            if r < ri:
                return 0
            elif ri <= r < Rsp:
                return self._rhoPrime(r,mBH,rhosMW,rs)/mx
            else:
                return rhox(r,rhosMW,rs)/mx
        else:
            Rsp = self._radiusSpike(mBH,rhosMW,rs)
            sigv = sigv*1e-26
            rhoc = mx/sigv/tBH/self.year2Seconds
            if r < ri:
                return 0
            elif ri <= r < Rsp:
                rhoP = self._rhoPrime(r,mBH,rhosMW,rs)
                return rhoP*rhoc/(rhoP + rhoc)/mx
            else:
                rhoDM = rhox(r,rhosMW,rs)
                return rhoDM*rhoc/(rhoDM + rhoc)/mx
    
    def __call__(self,r,mx,MG,sigv,tBH,rhosMW,rsMW,eta):
        return self._nxSpike(r,mx,MG,sigv,tBH,rhosMW,rsMW,eta)


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
    return (MG/constant.Mmw)**(1/3)*rsMW


def nxNFW(r,mx,rhosMW=184,rsMW=24.42,MG=None) -> float:
    """
    DM number density at r for arbitrary MG
    
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
        rr = r/rs
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
    return 7e7*(eta*MG/1e12)**(4/3)


def radiusSchwarzchild(mBH) -> float:
    """
    Calculating the Schawarzchild radius 

    In
    ------
    mBH: Black hole mass, Msun

    Ou
    ------
    Rs: Schwarzchild radius, kpc
    """
    mBH = mBH*constant.Msun_kg
    Rs = mBH*1.48e-25/constant.kpc2cm  # convert Rs into kpc
    return Rs


def dmNumberDensity(r,mx,MG,is_spike=True,sigv=None,tBH=1e9,rhosMW=184,rsMW=24.42,eta=24.3856) -> float:
    """
    Obtain the DM number density at given r with arbitrary MG
    
    In
    ------
    r: distance to GC, kpc
    mx: DM mass, MeV
    MG: The galactic stellar mass, Msun
    is_spike: Turn on/off spike feature, bool
    sigv: DM annihilation cross section, in the unit of cm^-3
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
        return nxNFW(r,mx,rhosMW,rsMW,MG,eta)
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


def dsigma0(Ev,thetaCM) -> float:
    """
    Differential DM-neutrino scattering cross section in CM frame
    
    In
    ------
    Ev: incoming SN neutrino energy, MeV
    thetaCM: Scattering angle in CM frame, rad
    
    Out
    ------
    dsigma0: cm^2 per steradian
    """
    # this is energy-independent cross section
    # divided by 4*np implying isotropic in CM frame
    return 1e-35/4/_np.pi


def supernovaNuFlux(Ev,l) -> float:
    """
    SN neutrino flux after propagating a distance l
    
    Input
    ------
    Ev: SN neutrino energy, MeV
    l: propagation distance, kpc
    
    Output
    ------
    flux: #/Ev/cm^2/s
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
    return L*(nue_dist + nueb_dist + 4*nux_dist)



# ----- Diffuse boosted dark matter -----

class dbdmSpectrum(constant):
    
    def __init__(self):
        pass
    
    def _diffSpectrum(self,Tx,mx,MG,R,l,theta,thetaCM,is_spike,sigv,tBH,rhosMW,rsMW,eta):
        """
        dNx/dTx
        """
        r = _get_r(l,R,theta)
        if r >= 1e-8:
            Ev = snNuEenergy(Tx,mx,thetaCM)
            dEvdTx = _dEv(Tx,mx,thetaCM)
            vx = vBDM(Tx,mx)  #  
            nx = dmNumberDensity(r,mx,MG,is_spike,sigv,tBH,rhosMW,rsMW,eta)
            return l**2*_np.sin(theta)*_np.sin(thetaCM)*nx*dsigma0(Ev,thetaCM)*supernovaNuFlux(Ev,l)*(dEvdTx*vx)
        else:
            return 0
    
    def _dbdmSpectrum(self,z,m,Tx,mx,R,l,theta,thetaCM,is_spike,sigv,tBH,rhosMW,rsMW,eta):
        """
        DBDM spectrume yielded by SN at arbitrary position R
        """
        Txp = (1 + z)*Tx 
        if Txp < 200:  # discard the BDM signature if it requires Ev > 200 MeV at z 
            MG = 10**m
            return MG*dnG(m,z)/_E(z)*rhoDotSFR(z)*self._diffSpectrum(Txp,mx,MG,R,l,theta,thetaCM,is_spike,sigv, \
                                                                    tBH,rhosMW,rsMW,eta)
        else:
            return 0
        
    def _dbdmSpectrumWeighted(self,z,m,Tx,mx,R,l,theta,thetaCM,is_spike,sigv,tBH,rhosMW,rsMW,eta,usefit):
        """
        DBDM spectrume yielded by SN at position R weighted by galactic baryonic distribution
        """
        Txp = (1 + z)*Tx 
        if Txp < 200:  # discard the BDM signature if it requires Ev > 200 MeV at z
            MG = 10**m
            # adopt fitting data for galactic area density?
            if usefit is True:
                galArealDensity = galacticAreaDensityFit((R,m))
            elif usefit is False:
                galArealDensity = galacticAreaDensity(R,zRange=[-10,10],MG=MG)
            else:
                raise FlagError('Global flag \'usefit\' must be a boolean.')
            
            return 2*_np.pi*R*galArealDensity*dnG(m,z)/_E(z)*rhoDotSFR(z)*self._diffSpectrum(Tx,mx,MG,R,l,theta,thetaCM, \
                                                                                           is_spike,sigv,tBH,rhosMW,rsMW,eta)
        else:
            return 0
    
    def __call__(self,z,m,Tx,mx,R,l,theta,thetaCM,is_spike,is_weighted,sigv,tBH,rhosMW,rsMW,eta,usefit):
        if is_weighted is True:
            return self._dbdmSpectrumWeighted(z,m,Tx,mx,R,l,theta,thetaCM,is_spike,sigv,tBH,rhosMW,rsMW,eta,usefit)
        elif is_weighted is False:
            return self._dbdmSpectrum(z,m,Tx,mx,R,l,theta,thetaCM,is_spike,sigv,tBH,rhosMW,rsMW,eta)
        else:
            raise FlagError('Flag \'is_weighted\' must be a boolean.')


def flux(Tx,mx,                                                           \
             R=0,Rmax=500,rmax=500,tau=10,is_spike=True,is_average=True,      \
             sigv=None,tBH=1e9,rhosMW=184,rsMW=24.42,eta=24.3856,usefit=True, \
             nitn=10,neval=50000):
    """
    DBDM flux for given (Tx,mx)
    
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
    sigv: DM annihilation cross section, in the unit of 3e-26 cm^3/s
        None indicates no annihilation
    tBH: BH age
    rhosMW: NFW characteristic density for MW
    rsMW: NFW characteristic radius for MW
    eta: Mmw/Mhalo for MW
    nitn: Number of chains in vegas
    neval: Number of evaluation in each MCMC chain
    
    Out
    ------
    Flux: per MeV per cm^2 per second
    """
    preFactor = constant.MagicalNumber #constant.D_H0*0.017/constant.Mmw/rhoDotSFR(0)/1e6/constant.kpc2cm**2/constant.year2Seconds
    lmax = Rmax + rmax
    spectrum  = dbdmSpectrum()
    if is_average is True:
        integrator = _vegas.Integrator([[0,8],[6,12],[0,Rmax],[0,lmax],[0,_np.pi],[0,_np.pi]]) #(z,m,R,l,theta,thetaCM)
        result = integrator(lambda x: spectrum(z=x[0],m=x[1],Tx=Tx,mx=mx,R=x[2],l=x[3],theta=x[4],thetaCM=x[5], \
                                               is_spike=is_spike,is_weighted=is_average,sigv=sigv,tBH=tBH,      \
                                               rhosMW=rhosMW,rsMW=rsMW,eta=eta,usefit=usefit),nitn=nitn,neval=neval).mean
        flux = 4*_np.pi**2*tau*result*constant.kpc2cm**3*vBDM(Tx,mx)*preFactor
    elif is_average is False:
        integrator = _vegas.Integrator([[0,8],[6,12],[0,lmax],[0,_np.pi],[0,_np.pi]]) #(z,m,l,theta,thetaCM)
        result = integrator(lambda x: spectrum(z=x[0],m=x[1],Tx=Tx,mx=mx,R=R,l=x[2],theta=x[3],thetaCM=x[4],    \
                                               is_spike=is_spike,is_weighted=is_average,sigv=sigv,tBH=tBH,      \
                                               rhosMW=rhosMW,rsMW=rsMW,eta=eta,usefit=usefit),nitn=nitn,neval=neval).mean
        flux = 4*_np.pi**2*tau*result*constant.kpc2cm**3*vBDM(Tx,mx)*preFactor
    else:
        raise FlagError('Flag \'is_average\' must be a boolean.')
    return flux


def event(mx,                                                                          \
              TxRange=[5,100],R=0,Rmax=500,rmax=500,tau=10,is_spike=True,is_average=True,  \
              sigv=None,tBH=1e9,rhosMW=184,rsMW=24.42,eta=24.3856,usefit=True,             \
              nitn=10,neval=50000):
    """
    DBDM event for given mx
    
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
    sigv: DM annihilation cross section, in the unit of 3e-26 cm^3/s
        None indicates no annihilation
    tBH: BH age
    rhosMW: NFW characteristic density for MW
    rsMW: NFW characteristic radius for MW
    eta: Mmw/Mhalo for MW
    nitn: Number of chains in vegas
    neval: Number of evaluation in each MCMC chain
    
    Out
    ------
    Event: per cm^2 per second
    """
    preFactor = constant.MagicalNumber #constant.D_H0*0.017/constant.Mmw/rhoDotSFR(0)/1e6/constant.kpc2cm**2/constant.year2Seconds
    lmax = Rmax + rmax
    spectrum  = dbdmSpectrum()
    if is_average is True:
        integrator = _vegas.Integrator([[0,8],[6,12],[0,Rmax],[0,lmax],[0,_np.pi],[0,_np.pi],TxRange]) #(z,m,R,l,theta,thetaCM)
        result = integrator(lambda x: spectrum(z=x[0],m=x[1],Tx=x[6],mx=mx,R=x[2],l=x[3],theta=x[4],thetaCM=x[5],  \
                                               is_spike=is_spike,is_weighted=is_average,sigv=sigv,tBH=tBH,         \
                                               rhosMW=rhosMW,rsMW=rsMW,eta=eta,usefit=usefit)*vBDM(Tx=x[6],mx=mx), \
                            nitn=nitn,neval=neval).mean
        event = 4*_np.pi**2*tau*result*constant.kpc2cm**3*preFactor
    elif is_average is False:
        integrator = _vegas.Integrator([[0,8],[6,12],[0,lmax],[0,_np.pi],[0,_np.pi],TxRange]) #(z,m,l,theta,thetaCM,Tx)
        result = integrator(lambda x: spectrum(z=x[0],m=x[1],Tx=x[5],mx=mx,R=R,l=x[2],theta=x[3],thetaCM=x[4],     \
                                               is_spike=is_spike,is_weighted=is_average,sigv=sigv,tBH=tBH,         \
                                               rhosMW=rhosMW,rsMW=rsMW,eta=eta,usefit=usefit)*vBDM(Tx=x[5],mx=mx), \
                            nitn=nitn,neval=neval).mean
        event = 4*_np.pi**2*tau*result*constant.kpc2cm**3*preFactor
    else:
        raise FlagError('Flag \'is_average\' must be a boolean.')
    return event
