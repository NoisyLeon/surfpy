
import os
import numpy as np
import matplotlib.pyplot as plt
import copy

datadir='/home/leon/code/CPSPy'


class Model1d(object):
    """
    An object to handle input 1d model for Computer Programs in Seismology.
    ===========================================================================================================
    Parameters:
    modelver        - model version
    modelname       - model name
    modelindex      - index indicating model type
                        1: 'ISOTROPIC', 2: 'TRANSVERSE ISOTROPIC', 3: 'ANISOTROPIC'
    modelunit       - KGS km, km/s, g/cm^3
    earthindex      - index indicating Earth type 1: 'FLAT EARTH', 2:'SPHERICAL EARTH'
    boundaryindex   - index indicating model boundaries 1: '1-D', 2: '2-D', 3: '3-D'
    Vindex          - index indicating nature of layer velocity
    HArr            - layer thickness array
    VsArr, VpArr, rhoArr, QpArr, QsArr, etapArr, etasArr, frefpArr,  frefsArr
                    - model parameters (isotropic)
    VsvArr, VpvArr, VshArr, VphArr, VpfArr, rhoArr, QpArr, QsArr, etapArr, etasArr, frefpArr,  frefsArr
                    - model parameters (TI)                
    DepthArr        - depth array (bottom depth of each layer)
    ===========================================================================================================
    """
    def __init__(self, modelver='MODEL.01', modelname='TEST MODEL', modelindex=1, modelunit='KGS', earthindex=2,
            boundaryindex=1, Vindex=1, HArr=np.array([]), VsArr=np.array([]), VpArr=np.array([]), rhoArr=np.array([]),
            QpArr=np.array([]), QsArr=np.array([]), etapArr=np.array([]), etasArr=np.array([]), frefpArr=np.array([]),  frefsArr=np.array([])):
        self.modelver   = modelver
        self.modelname  = modelname
        modeldict       = {1: 'ISOTROPIC', 2: 'TRANSVERSE ISOTROPIC', 3: 'ANISOTROPIC'}
        if modelindex != 1 and modelindex != 2:
            raise ValueError('Error in model index, currently only support 1 : ISOTROPIC and 2: TRANSVERSE ISOTROPIC')
        self.modeltype  = modeldict[modelindex]
        self.modelunit  = modelunit
        earthdict       = {1: 'FLAT EARTH', 2:'SPHERICAL EARTH'}
        self.earthtype  = earthdict[earthindex]
        boundarydict    = {1: '1-D', 2: '2-D', 3: '3-D'}
        self.boundarytype=boundarydict[boundaryindex]
        Vdict           = {1: 'CONSTANT VELOCITY', 2: 'VARIABLE VELOCITY'}
        self.Vtype      = Vdict[Vindex]
        self.line08_11  = 'LINE08\nLINE09\nLINE10\nLINE11\n'
        if self.modeltype == 'ISOTROPIC':
            self.modelheader= '\tH(KM)\tVP(KM/S)\tVS(KM/S)\tRHO(GM/CC)\tQP\tQS\tETAP\tETAS\tFREFP\tFREFS'
        elif self.modeltype == 'TRANSVERSE ISOTROPIC':
            self.modelheader= '\tH(KM)\tVPV(KM/S)\tVSV(KM/S)\tRHO(GM/CC)\tQP\tQS\tETAP\tETAS\tFREFP\tFREFS\n\tVPH(KM/S)\tVSH(KM/S)\tVPF(KM/S)'
        self.HArr       = HArr
        if self.modeltype == 'ISOTROPIC':
            self.VsArr      = VsArr
            self.VpArr      = VpArr
        elif self.modeltype == 'TRANSVERSE ISOTROPIC':
            self.VsvArr     = VsArr
            self.VpvArr     = VpArr
            self.VshArr     = VsArr
            self.VphArr     = VpArr
            if VsArr.size == 0 or VpArr.size==0 or VsArr.size != VpArr.size:
                self.VpfArr = np.array([])
            else:
                self.VpfArr = np.sqrt(VpArr**2 - 2*VsArr**2)
        self.rhoArr     = rhoArr
        self.QpArr      = QpArr
        self.QsArr      = QsArr
        self.etapArr    = etapArr
        self.etasArr    = etasArr
        self.frefpArr   = frefpArr
        self.frefsArr   = frefsArr
        self.DepthArr   = np.cumsum(self.HArr)
        return
    
    def copy(self): return copy.deepcopy(self)
    
    def ak135(self, modelname='AK135 CONTINENTAL MODEL'):
        """
        Load ak135 model
        """
        self.modelname  = modelname
        if os.path.isfile(datadir+'/ak135_dbase.txt'):
            ak135Arr        = np.loadtxt(datadir+'/ak135_dbase.txt')
        else:
            ak135Arr        = np.loadtxt('ak135_dbase.txt')
        self.HArr       = ak135Arr[:,0]
        if self.modeltype == 'ISOTROPIC':
            self.VpArr      = ak135Arr[:,1]
            self.VsArr      = ak135Arr[:,2]
        elif self.modeltype == 'TRANSVERSE ISOTROPIC':
            self.VsvArr     = ak135Arr[:,2]
            self.VpvArr     = ak135Arr[:,1]
            self.VshArr     = ak135Arr[:,2]
            self.VphArr     = ak135Arr[:,1]
            self.VpfArr     = np.sqrt ( (ak135Arr[:,1])**2 - 2.*((ak135Arr[:,2])**2) )
        self.rhoArr     = ak135Arr[:,3]
        self.QpArr      = ak135Arr[:,4]
        self.QsArr      = ak135Arr[:,5]
        self.etapArr    = ak135Arr[:,6]
        self.etasArr    = ak135Arr[:,7]
        self.frefpArr   = ak135Arr[:,8]
        self.frefsArr   = ak135Arr[:,9]
        self.DepthArr   = np.cumsum(self.HArr)
        self.check_model(verbose=False, trim=True)
        return
    
    def getmodel(self, modelname, HArr, VpArr, VsArr, rhoArr, QpArr, QsArr, etap=0., etas=0., frefp=1.0, fres=1.):
        """
        get model
        """
        self.modelname  = modelname
        self.HArr       = HArr
        if self.modeltype == 'ISOTROPIC':
            self.VsArr      = VsArr
            self.VpArr      = VpArr
        elif self.modeltype == 'TRANSVERSE ISOTROPIC':
            self.VsvArr     = VsArr
            self.VpvArr     = VpArr
            self.VshArr     = VsArr
            self.VphArr     = VpArr
            self.VpfArr     = np.sqrt(VpArr**2 - 2.*(VsArr**2))
        self.rhoArr     = rhoArr
        self.QpArr      = QpArr
        self.QsArr      = QsArr
        self.etapArr    = etap*np.ones(HArr.size)
        self.etasArr    = etas*np.ones(HArr.size)
        self.frefpArr   = frefp*np.ones(HArr.size)
        self.frefsArr   = fres*np.ones(HArr.size)
        self.DepthArr   = np.cumsum(self.HArr)
        return
    
    def trim(self, ind0=None, indf=None, zmax=None):
        """
        Trime the model with given index ind0, indf and maximum depth zmax
        """
        add_last_layer  = False
        if zmax != None and indf == None:
            index = (np.where(self.DepthArr> zmax)[0])
            if index.size != 0:
                indf = index[0]
                if self.DepthArr[indf-1] < zmax:
                    add_last_layer = True
                    H       = zmax - self.DepthArr[indf-1]
                    if self.modeltype == 'ISOTROPIC':
                        vs  = self.VsArr[indf]
                        vp  = self.VpArr[indf]
                    elif self.modeltype == 'TRANSVERSE ISOTROPIC':
                        vsv = self.VsvArr[indf]
                        vpv = self.VpvArr[indf]
                        vsh = self.VshArr[indf]
                        vph = self.VphArr[indf]
                        vpf = self.VpfArr[indf]
                    rho     = self.rhoArr[indf]
                    Qp      = self.QpArr[indf]
                    Qs      = self.QsArr[indf]
                    etap    = self.etapArr[indf]
                    etas    = self.etasArr[indf]
                    frefp   = self.frefpArr[indf]
                    frefs   = self.frefsArr[indf]
        self.HArr       = self.HArr[ind0:indf]
        if self.modeltype == 'ISOTROPIC':
            self.VpArr      = self.VpArr[ind0:indf]
            self.VsArr      = self.VsArr[ind0:indf]
        elif self.modeltype == 'TRANSVERSE ISOTROPIC':
            self.VsvArr     = self.VsvArr[ind0:indf]
            self.VpvArr     = self.VpvArr[ind0:indf]
            self.VshArr     = self.VshArr[ind0:indf]
            self.VphArr     = self.VphArr[ind0:indf]
            self.VpfArr     = self.VpfArr[ind0:indf]
        self.rhoArr     = self.rhoArr[ind0:indf]
        self.QpArr      = self.QpArr[ind0:indf]
        self.QsArr      = self.QsArr[ind0:indf]
        self.etapArr    = self.etapArr[ind0:indf]
        self.etasArr    = self.etasArr[ind0:indf]
        self.frefpArr   = self.frefpArr[ind0:indf]
        self.frefsArr   = self.frefsArr[ind0:indf]
        self.DepthArr   = self.DepthArr[ind0:indf]
        if add_last_layer:
            if self.modeltype == 'ISOTROPIC':
                self.addlayer(H=H, vsv=vs, vsh=vs, vpv=vp, vph=vp, rho=rho,
                                    Qp=Qp, Qs=Qs, etap=etap, etas=etas, frefp=frefp, frefs=frefs)
            if self.modeltype == 'TRANSVERSE ISOTROPIC':
                self.addlayer(H=H, vsv=vsv, vsh=vsh, vpv=vpv, vph=vph, vpf=vpf, rho=rho,
                                    Qp=Qp, Qs=Qs, etap=etap, etas=etas, frefp=frefp, frefs=frefs)
        if self.HArr.size == 0: print 'WARNING: trimed model has a length of zero!'
        return
    
    def relayerize(self, h, npinterp=False):
        """
        Re-laterize the model with given depth spacing
        """
        tmodel      = self.copy()
        zmax        = self.DepthArr[-1]
        if (zmax/h)%1 > 1e-5: print 'WARNING: zmax is not integer multiple of layer thickness!'
        tmodel.HArr        = np.ones(int(np.floor(zmax/h)), dtype=float)*h
        if npinterp:
            tmodel.DepthArr = np.cumsum(tmodel.HArr)
            midzArr         = tmodel.DepthArr - h/2.
            z0Arr           = self.DepthArr-self.HArr/2.
            if tmodel.modeltype == 'ISOTROPIC':
                tmodel.VsArr    = np.interp(midzArr, xp = z0Arr, fp = self.VsArr)
                tmodel.VpArr    = np.interp(midzArr, xp = z0Arr, fp = self.VpArr)
            elif tmodel.modeltype == 'TRANSVERSE ISOTROPIC':
                tmodel.VsvArr   = np.interp(midzArr, xp = z0Arr, fp = self.VsvArr)
                tmodel.VpvArr   = np.interp(midzArr, xp = z0Arr, fp = self.VpvArr)
                tmodel.VshArr   = np.interp(midzArr, xp = z0Arr, fp = self.VshArr)
                tmodel.VphArr   = np.interp(midzArr, xp = z0Arr, fp = self.VphArr)
                tmodel.VpfArr   = np.interp(midzArr, xp = z0Arr, fp = self.VpfArr)
            tmodel.rhoArr       = np.interp(midzArr, xp = z0Arr, fp = self.rhoArr)
            tmodel.QpArr        = np.interp(midzArr, xp = z0Arr, fp = self.QpArr)
            tmodel.QsArr        = np.interp(midzArr, xp = z0Arr, fp = self.QsArr)
            tmodel.etapArr      = np.interp(midzArr, xp = z0Arr, fp = self.etapArr)
            tmodel.etasArr      = np.interp(midzArr, xp = z0Arr, fp = self.etasArr)
            tmodel.frefpArr     = np.interp(midzArr, xp = z0Arr, fp = self.frefpArr)
            tmodel.frefsArr     = np.interp(midzArr, xp = z0Arr, fp = self.frefsArr)
        else:
            if tmodel.modeltype == 'ISOTROPIC':
                VsArr   = np.array([])
                VpArr   = np.array([])
            elif tmodel.modeltype == 'TRANSVERSE ISOTROPIC':
                VsvArr  = np.array([])
                VpvArr  = np.array([])
                VshArr  = np.array([])
                VphArr  = np.array([])
                VpfArr  = np.array([])
            rhoArr      = np.array([])
            QpArr       = np.array([])
            QsArr       = np.array([])
            etapArr     = np.array([])
            etasArr     = np.array([])
            frefpArr    = np.array([])
            frefsArr    = np.array([])
            zbArr       = np.cumsum(tmodel.HArr)
            for zb in zbArr:
                zt      = zb -h
                indt    = (np.where(zt <= self.DepthArr)[0])[0] 
                # if indt < 0: indt = 0
                indb    = (np.where(zb <= self.DepthArr)[0])[0] 
                # if indb < 0: indb = 0
                if tmodel.modeltype == 'ISOTROPIC':
                    VsArr   = np.append(VsArr, (self.VsArr[indt]+self.VsArr[indb])/2.)
                    VpArr   = np.append(VpArr, (self.VpArr[indt]+self.VpArr[indb])/2.)
                elif tmodel.modeltype == 'TRANSVERSE ISOTROPIC':
                    VsvArr  = np.append(VsvArr, (self.VsvArr[indt]+self.VsvArr[indb])/2.)
                    VpvArr  = np.append(VpvArr, (self.VpvArr[indt]+self.VpvArr[indb])/2.)
                    VshArr  = np.append(VshArr, (self.VshArr[indt]+self.VshArr[indb])/2.)
                    VphArr  = np.append(VphArr, (self.VphArr[indt]+self.VphArr[indb])/2.)
                    VpfArr  = np.append(VpfArr, (self.VpfArr[indt]+self.VpfArr[indb])/2.)
                rhoArr      = np.append(rhoArr, (self.rhoArr[indt]+self.rhoArr[indb])/2.)
                QpArr       = np.append(QpArr, (self.QpArr[indt]+self.QpArr[indb])/2.)
                QsArr       = np.append(QsArr, (self.QsArr[indt]+self.QsArr[indb])/2.)
                etapArr     = np.append(etapArr, (self.etapArr[indt]+self.etapArr[indb])/2.)
                etasArr     = np.append(etasArr, (self.etasArr[indt]+self.etasArr[indb])/2.)
                frefpArr    = np.append(frefpArr, (self.frefpArr[indt]+self.frefpArr[indb])/2.)
                frefsArr    = np.append(frefsArr, (self.frefsArr[indt]+self.frefsArr[indb])/2.)
            if tmodel.modeltype == 'ISOTROPIC':
                tmodel.VsArr    = VsArr
                tmodel.VpArr    = VpArr
            elif tmodel.modeltype == 'TRANSVERSE ISOTROPIC':
                tmodel.VsvArr   = VsvArr
                tmodel.VpvArr   = VpvArr
                tmodel.VshArr   = VshArr
                tmodel.VphArr   = VphArr
                tmodel.VpfArr   = VpfArr
            tmodel.rhoArr       = rhoArr
            tmodel.QpArr        = QpArr
            tmodel.QsArr        = QsArr
            tmodel.etapArr      = etapArr
            tmodel.etasArr      = etasArr
            tmodel.frefpArr     = frefpArr
            tmodel.frefsArr     = frefsArr
            tmodel.DepthArr     = zbArr
        return tmodel
    
    def check_iso_model(self):
        if self.modeltype != 'ISOTROPIC': raise ValueError('check_iso_model function only works for isotropic model!')
        if (self.HArr[self.HArr<1.0]).size %2 !=0: raise ValueError('Invalid vertical profile! Check layer thicknesses!')
        tempH   = self.HArr[self.HArr<1.0]
        tempVs  = self.VsArr[self.HArr<1.0]
        tempVp  = self.VpArr[self.HArr<1.0]
        temprho = self.rhoArr[self.HArr<1.0]
        tempQs  = self.QsArr[self.HArr<1.0]
        tempQp  = self.QpArr[self.HArr<1.0]
        ind0    = np.arange(tempH.size/2, dtype=int)*2
        ind1    = ind0 + 1
        HArr    = np.append( (tempH[ind0] + tempH[ind1]), self.HArr[self.HArr>=1.0] )
        VsArr   = np.append( (tempVs[ind0] + tempVs[ind1])/2., self.VsArr[self.HArr>=1.0] )
        VpArr   = np.append( (tempVp[ind0] + tempVp[ind1])/2., self.VpArr[self.HArr>=1.0] )
        rhoArr  = np.append( (temprho[ind0] + temprho[ind1])/2., self.rhoArr[self.HArr>=1.0] )
        QsArr   = np.append( (tempQs[ind0] + tempQs[ind1])/2., self.QsArr[self.HArr>=1.0] )
        QpArr   = np.append( (tempQp[ind0] + tempQp[ind1])/2., self.QpArr[self.HArr>=1.0] )
        if HArr.size <= 200:
            self.getmodel(modelname=self.modelname, HArr=HArr, VpArr=VpArr, VsArr=VsArr,
                        rhoArr=rhoArr, QpArr=QpArr, QsArr=QsArr)
        else:
            self.getmodel(modelname=self.modelname, HArr=HArr[:200], VpArr=VpArr[:200], VsArr=VsArr[:200],
                        rhoArr=rhoArr[:200], QpArr=QpArr[:200], QsArr=QsArr[:200])
        return
    
    def check_model(self, verbose=True, trim=False):
        """
        Check model validity. Currently zero values of vsv/vsh for non-top layer is not allowed!
        """
        if self.modeltype == 'TRANSVERSE ISOTROPIC':
            indexsv     = np.where(self.VsvArr < 0.001)[0]
            indexsh     = np.where(self.VshArr < 0.001)[0]
            if indexsv.size != indexsh.size:
                raise ValueError('Incompatible zero values for Vsv and Vsh of TI model!')
            if not np.allclose(indexsv, indexsh):
                raise ValueError('Incompatible zero values for Vsv and Vsh of TI model!')
            if indexsv.size == 0: return True
            elif indexsv.size == 1 and indexsv[0] == 0: return True
            else:
                if verbose: print 'WARNING: For TI model, only the top Vsv/Vsh can be zero!'
                if trim:
                    if verbose: print 'Model will be trimed to discard zero non-top Vsv/Vsh!'
                    if indexsv[0] == 0:
                        self.trim(indf=indexsv[1])
                    else:
                        self.trim(indf=indexsv[0])
        return
    

    def addlayer(self, H, vsv, vsh=None, vpv=None, vph=None, vpf=None, rho=None,
                Qp=310., Qs=150., etap=0.0, etas=0.0, frefp=1.0, frefs=1.0, zmin=9999.):
        """ Add layer to the model
        =========================================================================================================================
        Input Parameters:
        H               - layer thickness
        vsv, vsh        - SV/SH velocity
        vpv, vph        - PV/PH velocity
        vpf             - constructed velocity related to F modulus
        rho             - density
                            default is None by assuming Brocher Crust
        Qp, Qs          - quality factor
        etap, etas, frefp, frefs
                        - see the manual for computer programs in seismology
        zmin            -   top depth of the layer
                            1. default is 9999, which will simply append one layer to the model
                            2. if zmin < the bottom of preexisting top layer, it will append a new
                                layer to the top (also replace some part of the preexisting top layer)
                            3. else, the code will replace some part of preexisting model( this part need further checking! )
        =========================================================================================================================
        """
        #####
        # Default: Brocher crust
        #####
        if vpv  is None: vpv    = 0.9409+2.0947*vsv-0.8206*vsv**2+0.2683*vsv**3-0.0251*vsv**4
        if vph  is None: vph    = vpv
        if vsh  is None: vsh    = vsv
        if rho  is None: rho    = 1.6612*vpv-0.4721*vpv**2+0.0671*vpv**3-0.0043*vpv**4+0.000106*vpv**5
        if vpf  is None: vpf    = np.sqrt(vpv**2 - 2.*vsv**2)
        if self.HArr.size==0:
            self.HArr       = np.append(self.HArr, H)
            if self.modeltype == 'ISOTROPIC':
                self.VpArr  = np.append(self.VpArr, vpv)
                self.VsArr  = np.append(self.VsArr, vsv)
            elif self.modeltype == 'TRANSVERSE ISOTROPIC':
                self.VpvArr = np.append(self.VpvArr, vpv)
                self.VsvArr = np.append(self.VsvArr, vsv)
                self.VphArr = np.append(self.VphArr, vph)
                self.VshArr = np.append(self.VshArr, vsh)
                self.VpfArr = np.append(self.VpfArr, vpf)
            self.rhoArr     = np.append(self.rhoArr, rho)
            self.QpArr      = np.append(self.QpArr, Qp)
            self.QsArr      = np.append(self.QsArr, Qs)
            self.etapArr    = np.append(self.etapArr, etap)
            self.etasArr    = np.append(self.etasArr, etas)
            self.frefpArr   = np.append(self.frefpArr, frefp)
            self.frefsArr   = np.append(self.frefsArr, frefs)
            self.DepthArr   = np.cumsum(self.HArr)
            return
        if zmin >= self.DepthArr[-1]:
            self.HArr       = np.append(self.HArr, H)
            if self.modeltype == 'ISOTROPIC':
                self.VpArr  = np.append(self.VpArr, vpv)
                self.VsArr  = np.append(self.VsArr, vsv)
            elif self.modeltype == 'TRANSVERSE ISOTROPIC':
                self.VpvArr = np.append(self.VpvArr, vpv)
                self.VsvArr = np.append(self.VsvArr, vsv)
                self.VphArr = np.append(self.VphArr, vph)
                self.VshArr = np.append(self.VshArr, vsh)
                self.VpfArr = np.append(self.VpfArr, vpf)
            self.rhoArr     = np.append(self.rhoArr, rho)
            self.QpArr      = np.append(self.QpArr, Qp)
            self.QsArr      = np.append(self.QsArr, Qs)
            self.etapArr    = np.append(self.etapArr, etap)
            self.etasArr    = np.append(self.etasArr, etas)
            self.frefpArr   = np.append(self.frefpArr, frefp)
            self.frefsArr   = np.append(self.frefsArr, frefs)
            self.DepthArr   = np.cumsum(self.HArr)
        elif zmin <= 0.:
            # discard layers with depth < H
            self.HArr       = self.HArr[self.DepthArr >H]
            if self.modeltype == 'ISOTROPIC':
                self.VpArr      = self.VpArr[self.DepthArr >H]
                self.VsArr      = self.VsArr[self.DepthArr >H]
            elif self.modeltype == 'TRANSVERSE ISOTROPIC':
                self.VpvArr     = self.VpvArr[self.DepthArr >H]
                self.VsvArr     = self.VsvArr[self.DepthArr >H]
                self.VphArr     = self.VphArr[self.DepthArr >H]
                self.VshArr     = self.VshArr[self.DepthArr >H]
                self.VpfArr     = self.VpfArr[self.DepthArr >H]
            self.rhoArr     = self.rhoArr[self.DepthArr >H]
            self.QpArr      = self.QpArr[self.DepthArr >H]
            self.QsArr      = self.QsArr[self.DepthArr >H]
            self.etapArr    = self.etapArr[self.DepthArr >H]
            self.etasArr    = self.etasArr[self.DepthArr >H]
            self.frefpArr   = self.frefpArr[self.DepthArr >H]
            self.frefsArr   = self.frefsArr[self.DepthArr >H]
            # change the thickness of the current first layer
            self.HArr[0]    = (self.DepthArr[self.DepthArr >H])[0]-H
            # add H to the first layer
            self.HArr       = np.append(H, self.HArr)
            if self.modeltype == 'ISOTROPIC':
                self.VpArr      = np.append(vpv, self.VpArr)
                self.VsArr      = np.append(vsv, self.VsArr)
            elif self.modeltype == 'TRANSVERSE ISOTROPIC':
                self.VpvArr     = np.append(vpv, self.VpvArr)
                self.VsvArr     = np.append(vsv, self.VsvArr)
                self.VphArr     = np.append(vph, self.VphArr)
                self.VshArr     = np.append(vsh, self.VshArr)
                self.VpfArr     = np.append(vpf, self.VpfArr)
            self.rhoArr     = np.append(rho, self.rhoArr)
            self.QpArr      = np.append(Qp, self.QpArr)
            self.QsArr      = np.append(Qs, self.QsArr)
            self.etapArr    = np.append(etap, self.etapArr)
            self.etasArr    = np.append(etas, self.etasArr)
            self.frefpArr   = np.append(frefp, self.frefpArr)
            self.frefsArr   = np.append(frefs, self.frefsArr)
            self.DepthArr   = np.cumsum(self.HArr)
        else:
            zmax        = zmin+H
            topArr      = self.DepthArr-self.HArr
            # Top layers above zmin
            HArrT       = self.HArr[self.DepthArr <=zmin]
            if self.modeltype == 'ISOTROPIC':
                VpArrT      = self.VpArr[self.DepthArr <=zmin]
                VsArrT      = self.VsArr[self.DepthArr <=zmin]
            elif self.modeltype == 'TRANSVERSE ISOTROPIC':
                VpvArrT     = self.VpvArr[self.DepthArr <=zmin]
                VsvArrT     = self.VsvArr[self.DepthArr <=zmin]
                VphArrT     = self.VphArr[self.DepthArr <=zmin]
                VshArrT     = self.VshArr[self.DepthArr <=zmin]
                VpfArrT     = self.VpfArr[self.DepthArr <=zmin]
            rhoArrT     = self.rhoArr[self.DepthArr <=zmin]
            QpArrT      = self.QpArr[self.DepthArr <=zmin]
            QsArrT      = self.QsArr[self.DepthArr <=zmin]
            etapArrT    = self.etapArr[self.DepthArr <=zmin]
            etasArrT    = self.etasArr[self.DepthArr <=zmin]
            frefpArrT   = self.frefpArr[self.DepthArr <=zmin]
            frefsArrT   = self.frefsArr[self.DepthArr <=zmin]
            tH          = zmin - (topArr[self.DepthArr >zmin]) [0]
            if tH != 0:
                if self.modeltype == 'ISOTROPIC':
                    tVp         = (self.VpArr[self.DepthArr >zmin])[0]
                    tVs         = (self.VsArr[self.DepthArr >zmin])[0]
                elif self.modeltype == 'TRANSVERSE ISOTROPIC':
                    tVpv        = (self.VpvArr[self.DepthArr >zmin])[0]
                    tVsv        = (self.VsvArr[self.DepthArr >zmin])[0]
                    tVph        = (self.VphArr[self.DepthArr >zmin])[0]
                    tVsh        = (self.VshArr[self.DepthArr >zmin])[0]
                    tVpf        = (self.VpfArr[self.DepthArr >zmin])[0]
                trho        = (self.rhoArr[self.DepthArr >zmin])[0]
                tQp         = (self.QpArr[self.DepthArr >zmin])[0]
                tQs         = (self.QsArr[self.DepthArr >zmin])[0]
                tetap       = (self.etapArr[self.DepthArr >zmin])[0]
                tetas       = (self.etasArr[self.DepthArr >zmin])[0]
                tfrefp      = (self.frefpArr[self.DepthArr >zmin])[0]
                tfrefs      = (self.frefsArr[self.DepthArr >zmin])[0]
                HArrT       = np.append(HArrT, tH)
                if self.modeltype == 'ISOTROPIC':
                    VpArrT      = np.append(VpArrT, tVp)
                    VsArrT      = np.append(VsArrT, tVs)
                elif self.modeltype == 'TRANSVERSE ISOTROPIC':
                    VpvArrT     = np.append(VpvArrT, tVpv)
                    VsvArrT     = np.append(VsvArrT, tVsv)
                    VphArrT     = np.append(VphArrT, tVph)
                    VshArrT     = np.append(VshArrT, tVsh)
                    VpfArrT     = np.append(VpfArrT, tVpf)
                rhoArrT     = np.append(rhoArrT, trho)
                QpArrT      = np.append(QpArrT, tQp)
                QsArrT      = np.append(QsArrT, tQs)
                etapArrT    = np.append(etapArrT, tetap)
                etasArrT    = np.append(etasArrT, tetas)
                frefpArrT   = np.append(frefpArrT, tfrefp)
                frefsArrT   = np.append(frefsArrT, tfrefs)
            # Bottom layer bolow zmax
            HArrB       = self.HArr[topArr >=zmax]
            if self.modeltype == 'ISOTROPIC':
                VpArrB      = self.VpArr[topArr >=zmax]
                VsArrB      = self.VsArr[topArr >=zmax]
            elif self.modeltype == 'TRANSVERSE ISOTROPIC':
                VpvArrB     = self.VpvArr[topArr >=zmax]
                VsvArrB     = self.VsvArr[topArr >=zmax]
                VphArrB     = self.VphArr[topArr >=zmax]
                VshArrB     = self.VshArr[topArr >=zmax]
                VpfArrB     = self.VpfArr[topArr >=zmax]
            rhoArrB     = self.rhoArr[topArr >=zmax]
            QpArrB      = self.QpArr[topArr >=zmax]
            QsArrB      = self.QsArr[topArr >=zmax]
            etapArrB    = self.etapArr[topArr >=zmax]
            etasArrB    = self.etasArr[topArr >=zmax]
            frefpArrB   = self.frefpArr[topArr >=zmax]
            frefsArrB   = self.frefsArr[topArr >=zmax]
            bH          = (self.DepthArr[topArr <zmax]) [-1]- zmax
            if bH != 0:
                if self.modeltype == 'ISOTROPIC':
                    bVp         = (self.VpArr[topArr <zmax]) [-1]
                    bVs         = (self.VsArr[topArr <zmax]) [-1]
                elif self.modeltype == 'TRANSVERSE ISOTROPIC':
                    bVpv        = (self.VpvArr[topArr <zmax]) [-1]
                    bVsv        = (self.VsvArr[topArr <zmax]) [-1]
                    bVph        = (self.VphArr[topArr <zmax]) [-1]
                    bVsh        = (self.VshArr[topArr <zmax]) [-1]
                    bVpf        = (self.VpfArr[topArr <zmax]) [-1]
                brho        = (self.rhoArr[topArr <zmax]) [-1]
                bQp         = (self.QpArr[topArr <zmax]) [-1]
                bQs         = (self.QsArr[topArr <zmax]) [-1]
                betap       = (self.etapArr[topArr <zmax]) [-1]
                betas       = (self.etasArr[topArr <zmax]) [-1]
                bfrefp      = (self.frefpArr[topArr <zmax]) [-1]
                bfrefs      = (self.frefsArr[topArr <zmax]) [-1]
                HArrB       = np.append(bH, HArrB)
                if self.modeltype == 'ISOTROPIC':
                    VpArrB      = np.append(bVp, VpArrB)
                    VsArrB      = np.append(bVs, VsArrB)
                elif self.modeltype == 'TRANSVERSE ISOTROPIC':
                    VpvArrB     = np.append(bVpv, VpvArrB)
                    VsvArrB     = np.append(bVsv, VsvArrB)
                    VphArrB     = np.append(bVph, VphArrB)
                    VshArrB     = np.append(bVsh, VshArrB)
                    VpfArrB     = np.append(bVpf, VpfArrB)
                rhoArrB     = np.append(brho, rhoArrB)
                QpArrB      = np.append(bQp, QpArrB)
                QsArrB      = np.append(bQs, QsArrB)
                etapArrB    = np.append(betap, etapArrB)
                etasArrB    = np.append(betas, etasArrB)
                frefpArrB   = np.append(bfrefp, frefpArrB)
                frefsArrB   = np.append(bfrefs, frefsArrB)
            #####################################
            self.HArr       = np.append(HArrT, H)
            if self.modeltype == 'ISOTROPIC':
                self.VpArr      = np.append(VpArrT, vpv)
                self.VsArr      = np.append(VsArrT, vsv)
            elif self.modeltype == 'TRANSVERSE ISOTROPIC':
                self.VpvArr     = np.append(VpvArrT, vpv)
                self.VsvArr     = np.append(VsvArrT, vsv)
                self.VphArr     = np.append(VphArrT, vph)
                self.VshArr     = np.append(VshArrT, vsh)
                self.VpfArr     = np.append(VpfArrT, vpf)
            self.rhoArr     = np.append(rhoArrT, rho)
            self.QpArr      = np.append(QpArrT, Qp)
            self.QsArr      = np.append(QsArrT, Qs)
            self.etapArr    = np.append(etapArrT, etap)
            self.etasArr    = np.append(etasArrT, etas)
            self.frefpArr   = np.append(frefpArrT, frefp)
            self.frefsArr   = np.append(frefsArrT,frefs)
            #####################################
            self.HArr       = np.append(self.HArr, HArrB)
            if self.modeltype == 'ISOTROPIC':
                self.VpArr      = np.append(self.VpArr, VpArrB)
                self.VsArr      = np.append(self.VsArr, VsArrB)
            elif self.modeltype == 'TRANSVERSE ISOTROPIC':
                self.VpvArr     = np.append(self.VpvArr, VpvArrB)
                self.VsvArr     = np.append(self.VsvArr, VsvArrB)
                self.VphArr     = np.append(self.VphArr, VphArrB)
                self.VshArr     = np.append(self.VshArr, VshArrB)
                self.VpfArr     = np.append(self.VpfArr, VpfArrB)
            self.rhoArr     = np.append(self.rhoArr, rhoArrB)
            self.QpArr      = np.append(self.QpArr, QpArrB)
            self.QsArr      = np.append(self.QsArr, QsArrB)
            self.etapArr    = np.append(self.etapArr, etapArrB)
            self.etasArr    = np.append(self.etasArr, etasArrB)
            self.frefpArr   = np.append(self.frefpArr, frefpArrB)
            self.frefsArr   = np.append(self.frefsArr, frefsArrB)
            self.DepthArr   = np.cumsum(self.HArr)
        return

    def write(self, outfname):
        """
        Write profile to the Computer Programs in Seismology model format
        """
        if self.modeltype == 'ISOTROPIC':
            outstr  = '%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n'
        elif self.modeltype == 'TRANSVERSE ISOTROPIC':
            outstr  = '%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n\t\t%f\t%f\t%f\n'
        with open(outfname, 'w') as f:
            f.write(self.modelver+'\n')
            f.write(self.modelname+'\n')
            f.write(self.modeltype+'\n')
            f.write(self.modelunit+'\n')
            f.write(self.earthtype+'\n')
            f.write(self.boundarytype+'\n')
            f.write(self.Vtype+'\n')
            f.write(self.line08_11)
            f.write(self.modelheader+'\n')
            for i in xrange(self.HArr.size):
                if self.modeltype == 'ISOTROPIC':
                    tempstr=outstr %(self.HArr[i], self.VpArr[i], self.VsArr[i], self.rhoArr[i],
                            self.QpArr[i], self.QsArr[i], self.etapArr[i], self.etasArr[i], self.frefpArr[i], self.frefsArr[i])
                elif self.modeltype == 'TRANSVERSE ISOTROPIC':
                    tempstr=outstr %(self.HArr[i], self.VpvArr[i], self.VsvArr[i], self.rhoArr[i], self.QpArr[i], self.QsArr[i],
                        self.etapArr[i], self.etasArr[i], self.frefpArr[i], self.frefsArr[i], self.VphArr[i], self.VshArr[i], self.VpfArr[i])
                f.write(tempstr)
        return
            
    def read(self, infname, verbose=True):
        """
        Read Computer Programs in Seismology model format
        """
        with open(infname, 'r') as f:
            self.modelver       = (f.readline()).split('\n')[0]
            self.modelname      = (f.readline()).split('\n')[0]
            modeltype      = (f.readline()).split('\n')[0]
            self.modelunit      = (f.readline()).split('\n')[0]
            self.earthtype      = (f.readline()).split('\n')[0]
            self.boundarytype   = (f.readline()).split('\n')[0]
            self.Vtype          = (f.readline()).split('\n')[0]
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            if modeltype == 'ISOTROPIC':
                self.modelheader    = (f.readline()).split('\n')[0]
                if self.modeltype == 'TRANSVERSE ISOTROPIC':
                    if verbose: print 'WARNING: reading isotropic mod file, modeltype is changed!'
                    self.VsArr          = np.array([])
                    self.VpArr          = np.array([])
                    del self.VsvArr, self.VpvArr, self.VshArr, self.VphArr, self.VpfArr
                    self.modeltype      = modeltype
            elif modeltype == 'TRANSVERSE ISOTROPIC':
                self.modelheader    = (f.readline()).split('\n')[0] + '\n'
                self.modelheader    += (f.readline()).split('\n')[0]
                if self.modeltype == 'ISOTROPIC':
                    if verbose: print 'WARNING: reading TI mod file, modeltype is changed!'
                    del self.VsArr, self.VpArr
                    self.modeltype      = modeltype
                    self.VsvArr         = np.array([])
                    self.VpvArr         = np.array([])
                    self.VshArr         = np.array([])
                    self.VphArr         = np.array([])
                    self.VpfArr         = np.array([])
            i=0
            for line in f.readlines():
                i               += 1
                cline           = line.split()
                if self.modeltype == 'ISOTROPIC':
                    self.HArr       = np.append(self.HArr, float(cline[0]))
                    self.VpArr      = np.append(self.VpArr, float(cline[1]))
                    self.VsArr      = np.append(self.VsArr, float(cline[2]))
                    self.rhoArr     = np.append(self.rhoArr, float(cline[3]))
                    self.QpArr      = np.append(self.QpArr, float(cline[4]))
                    self.QsArr      = np.append(self.QsArr, float(cline[5]))
                    self.etapArr    = np.append(self.etapArr, float(cline[6]))
                    self.etasArr    = np.append(self.etasArr, float(cline[7]))
                    self.frefpArr   = np.append(self.frefpArr, float(cline[8]))
                    self.frefsArr   = np.append(self.frefsArr, float(cline[9]))
                elif self.modeltype == 'TRANSVERSE ISOTROPIC':
                    if i%2 != 0:
                        self.HArr       = np.append(self.HArr, float(cline[0]))
                        self.VpvArr     = np.append(self.VpvArr, float(cline[1]))
                        self.VsvArr     = np.append(self.VsvArr, float(cline[2]))
                        self.rhoArr     = np.append(self.rhoArr, float(cline[3]))
                        self.QpArr      = np.append(self.QpArr, float(cline[4]))
                        self.QsArr      = np.append(self.QsArr, float(cline[5]))
                        self.etapArr    = np.append(self.etapArr, float(cline[6]))
                        self.etasArr    = np.append(self.etasArr, float(cline[7]))
                        self.frefpArr   = np.append(self.frefpArr, float(cline[8]))
                        self.frefsArr   = np.append(self.frefsArr, float(cline[9]))
                    else:
                        self.VphArr     = np.append(self.VphArr, float(cline[0]))
                        self.VshArr     = np.append(self.VshArr, float(cline[1]))
                        self.VpfArr     = np.append(self.VpfArr, float(cline[2]))
        return
    
    def read_layer_txt(self, infname):
        if self.modeltype != 'ISOTROPIC': raise ValueError('read_layer_txt function only works for isotropic model!')
        vpind=None; rhoind=None
        with open(infname) as f:
            inline = f.readline()
            if inline.split()[0] =='#':
                c=inline.split()[1:]
            c=np.array(c)
            z0ind = np.where(c=='z0')[0][0]
            Hind  = np.where(c=='H')[0][0]
            vsind = np.where(c=='vs')[0][0]
            try: vpind  = np.where(c=='vp')[0][0]
            except: pass
            try: rhoind = np.where(c=='rho')[0][0]
            except: pass
        inArr = np.loadtxt(infname)
        z0Arr = inArr[:, z0ind]
        HArr  = inArr[:, Hind]
        vsArr = inArr[:, vsind]
        # if vsind != None: vsArr = inArr[:, vsind]
        if vpind != None: vpArr = inArr[:, vpind]
        if rhoind != None: rhoArr = inArr[:, rhoind]
        for i in xrange(z0Arr.size):
            z0 = z0Arr[i]; H = HArr[i]; vs=vsArr[i]
            if vpind==None and rhoind ==None: self.addlayer( H=H, vs=vs, vp=None, rho=None, zmin=z0)
            elif vpind==None and rhoind !=None:
                rho=rhoArr[i]; self.addlayer( H=H, vs=vs, vp=None, rho=rho, zmin=z0)
            elif vpind!=None and rhoind ==None:
                vp=vpArr[i]; self.addlayer( H=H, vs=vs, vp=vp, rho=None, zmin=z0)
            elif vpind!=None and rhoind !=None:
                vp=vpArr[i]; rho=rhoArr[i]; self.addlayer( H=H, vs=vs, vp=vp, rho=rho, zmin=z0)
        return
    
    def perturb(self, dm, zmin=0, zmax=9999, datatype='vs'):
        """ Add perturbation to the model given a depth range
        =======================================================
        Input Parameters:
        dm          - perturbed value (-1, 1)
        zmin, zmax  - depth range for the perturbation
        datatype    - data type for perturbation
        =======================================================
        """
        if dm > 1. or dm < -1.: raise ValueError('Model parameter perturbation should be within [-1, 1] !')
        if (datatype == 'vs' or datatype == 'vp') and self.modeltype != 'ISOTROPIC':
            raise ValueError('datatype of vp/vs is not accepted when model is not isotropic !')
        if (datatype == 'vsv' or datatype == 'vpv' or datatype == 'vsh' or datatype == 'vph' or datatype == 'vpf') \
                and self.modeltype != 'TRANSVERSE ISOTROPIC':
            raise ValueError('datatype of vpv/vsv/vph/vsh is not accepted when model is not TI !')
        topArr      = self.DepthArr - self.HArr
        bottomArr   = self.DepthArr
        indexArr    = np.arange(self.HArr.size)
        indexT      = topArr >= zmin
        indexB      = bottomArr <= zmax
        if np.any(topArr == zmin): Ht = 0.
        else:
            print 'Will add top layer !'
            Ht      = ( topArr [topArr >zmin ]) [0] - zmin
            indext  = (indexArr [indexT])[0] - 1
            z1      = zmin
        if np.any(bottomArr == zmax): Hb = 0.
        else:
            print 'Will add bottom layer !'
            if zmax < bottomArr [0]:
                Hb      = zmax
                indexb  = 0
                z2      = 0.
            else:
                Hb      = zmax - ( bottomArr[bottomArr < zmax ]) [-1]
                indexb  = (indexArr [indexB])[-1] + 1
                z2      = ( bottomArr [bottomArr < zmax ]) [-1]
        index   = indexT*indexB
        if datatype == 'vp':
            self.VpArr[index]=self.VpArr[index]*(1.+dm)
            if Hb != 0:
                self.addlayer(H=Hb, vsv=self.VsArr[indexb], vpv=self.VpArr[indexb]*(1.+dm), rho=self.rhoArr[indexb],
                        Qp=self.QpArr[indexb], Qs=self.QsArr[indexb], etap=self.etapArr[indexb], etas=self.etasArr[indexb],
                            frefp=self.frefpArr[indexb], frefs=self.frefsArr[indexb], zmin=z2)
            if Ht != 0.:
                self.addlayer(H=Ht, vsv=self.VsArr[indext], vpv=self.VpArr[indext]*(1.+dm), rho=self.rhoArr[indext],
                        Qp=self.QpArr[indext], Qs=self.QsArr[indext], etap=self.etapArr[indext], etas=self.etasArr[indext],
                            frefp=self.frefpArr[indext], frefs=self.frefsArr[indext], zmin=z1)
                
        if datatype=='vs':
            self.VsArr[index]=self.VsArr[index]*(1.+dm)
            if Hb !=0:
                self.addlayer(H=Hb, vsv=self.VsArr[indexb]*(1.+dm), vpv=self.VpArr[indexb], rho=self.rhoArr[indexb],
                        Qp=self.QpArr[indexb], Qs=self.QsArr[indexb], etap=self.etapArr[indexb], etas=self.etasArr[indexb],
                            frefp=self.frefpArr[indexb], frefs=self.frefsArr[indexb], zmin=z2)
            if Ht !=0.:
                self.addlayer(H=Ht, vsv=self.VsArr[indext]*(1.+dm), vpv=self.VpArr[indext], rho=self.rhoArr[indext],
                        Qp=self.QpArr[indext], Qs=self.QsArr[indext], etap=self.etapArr[indext], etas=self.etasArr[indext],
                            frefp=self.frefpArr[indext], frefs=self.frefsArr[indext], zmin=z1)
                
        if datatype == 'vpv':
            self.VpvArr[index]=self.VpvArr[index]*(1.+dm)
            if Hb != 0:
                self.addlayer(H=Hb, vsv=self.VsvArr[indexb], vpv=self.VpvArr[indexb]*(1.+dm), vsh=self.VshArr[indexb],
                    vph=self.VphArr[indexb], vpf=self.VpfArr[indexb], rho=self.rhoArr[indexb],
                        Qp=self.QpArr[indexb], Qs=self.QsArr[indexb], etap=self.etapArr[indexb], etas=self.etasArr[indexb],
                            frefp=self.frefpArr[indexb], frefs=self.frefsArr[indexb], zmin=z2)
            if Ht != 0.:
                self.addlayer(H=Ht, vsv=self.VsvArr[indext], vpv=self.VpvArr[indext]*(1.+dm), vsh=self.VshArr[indext],
                    vph=self.VphArr[indext], vpf=self.VpfArr[indext], rho=self.rhoArr[indext],
                        Qp=self.QpArr[indext], Qs=self.QsArr[indext], etap=self.etapArr[indext], etas=self.etasArr[indext],
                            frefp=self.frefpArr[indext], frefs=self.frefsArr[indext], zmin=z1)
                
        if datatype == 'vph':
            self.VphArr[index]=self.VphArr[index]*(1.+dm)
            if Hb != 0:
                self.addlayer(H=Hb, vsv=self.VsvArr[indexb], vpv=self.VpvArr[indexb], vsh=self.VshArr[indexb],
                    vph=self.VphArr[indexb]*(1.+dm), vpf=self.VpfArr[indexb], rho=self.rhoArr[indexb],
                        Qp=self.QpArr[indexb], Qs=self.QsArr[indexb], etap=self.etapArr[indexb], etas=self.etasArr[indexb],
                            frefp=self.frefpArr[indexb], frefs=self.frefsArr[indexb], zmin=z2)
            if Ht != 0.:
                self.addlayer(H=Ht, vsv=self.VsvArr[indext], vpv=self.VpvArr[indext], vsh=self.VshArr[indext],
                    vph=self.VphArr[indext]*(1.+dm), vpf=self.VpfArr[indext], rho=self.rhoArr[indext],
                        Qp=self.QpArr[indext], Qs=self.QsArr[indext], etap=self.etapArr[indext], etas=self.etasArr[indext],
                            frefp=self.frefpArr[indext], frefs=self.frefsArr[indext], zmin=z1)
                
        if datatype == 'vsv':
            self.VsvArr[index]=self.VsvArr[index]*(1.+dm)
            if Hb != 0:
                self.addlayer(H=Hb, vsv=self.VsvArr[indexb]*(1.+dm), vpv=self.VpvArr[indexb], vsh=self.VshArr[indexb],
                    vph=self.VphArr[indexb], vpf=self.VpfArr[indexb], rho=self.rhoArr[indexb],
                        Qp=self.QpArr[indexb], Qs=self.QsArr[indexb], etap=self.etapArr[indexb], etas=self.etasArr[indexb],
                            frefp=self.frefpArr[indexb], frefs=self.frefsArr[indexb], zmin=z2)
            if Ht != 0.:
                self.addlayer(H=Ht, vsv=self.VsvArr[indext]*(1.+dm), vpv=self.VpvArr[indext], vsh=self.VshArr[indext],
                    vph=self.VphArr[indext], vpf=self.VpfArr[indext], rho=self.rhoArr[indext],
                        Qp=self.QpArr[indext], Qs=self.QsArr[indext], etap=self.etapArr[indext], etas=self.etasArr[indext],
                            frefp=self.frefpArr[indext], frefs=self.frefsArr[indext], zmin=z1)
                
        if datatype == 'vsh':
            self.VshArr[index]=self.VshArr[index]*(1.+dm)
            if Hb != 0:
                self.addlayer(H=Hb, vsv=self.VsvArr[indexb], vpv=self.VpvArr[indexb], vsh=self.VshArr[indexb]*(1.+dm),
                    vph=self.VphArr[indexb], vpf=self.VpfArr[indexb], rho=self.rhoArr[indexb],
                        Qp=self.QpArr[indexb], Qs=self.QsArr[indexb], etap=self.etapArr[indexb], etas=self.etasArr[indexb],
                            frefp=self.frefpArr[indexb], frefs=self.frefsArr[indexb], zmin=z2)
            if Ht != 0.:
                self.addlayer(H=Ht, vsv=self.VsvArr[indext], vpv=self.VpvArr[indext], vsh=self.VshArr[indext]*(1.+dm),
                    vph=self.VphArr[indext]*(1.+dm), vpf=self.VpfArr[indext], rho=self.rhoArr[indext],
                        Qp=self.QpArr[indext], Qs=self.QsArr[indext], etap=self.etapArr[indext], etas=self.etasArr[indext],
                            frefp=self.frefpArr[indext], frefs=self.frefsArr[indext], zmin=z1)
                
        if datatype == 'vpf':
            self.VpfArr[index]=self.VpfArr[index]*(1.+dm)
            if Hb != 0:
                self.addlayer(H=Hb, vsv=self.VsvArr[indexb], vpv=self.VpvArr[indexb], vsh=self.VshArr[indexb],
                    vph=self.VphArr[indexb], vpf=self.VpfArr[indexb]*(1.+dm), rho=self.rhoArr[indexb],
                        Qp=self.QpArr[indexb], Qs=self.QsArr[indexb], etap=self.etapArr[indexb], etas=self.etasArr[indexb],
                            frefp=self.frefpArr[indexb], frefs=self.frefsArr[indexb], zmin=z2)
            if Ht != 0.:
                self.addlayer(H=Ht, vsv=self.VsvArr[indext], vpv=self.VpvArr[indext], vsh=self.VshArr[indext],
                    vph=self.VphArr[indext], vpf=self.VpfArr[indext]*(1.+dm), rho=self.rhoArr[indext],
                        Qp=self.QpArr[indext], Qs=self.QsArr[indext], etap=self.etapArr[indext], etas=self.etasArr[indext],
                            frefp=self.frefpArr[indext], frefs=self.frefsArr[indext], zmin=z1)
        if self.modeltype == 'ISOTROPIC':
            if datatype=='rho':
                self.rhoArr[index]=self.rhoArr[index]*(1.+dm)
                if Hb !=0:
                    self.addlayer(H=Hb, vsv=self.VsArr[indexb], vpv=self.VpArr[indexb], rho=self.rhoArr[indexb]*(1.+dm),
                            Qp=self.QpArr[indexb], Qs=self.QsArr[indexb], etap=self.etapArr[indexb], etas=self.etasArr[indexb],
                                frefp=self.frefpArr[indexb], frefs=self.frefsArr[indexb], zmin=z2)
                if Ht !=0.:
                    self.addlayer(H=Ht, vsv=self.VsArr[indext], vpv=self.VpArr[indext], rho=self.rhoArr[indext]*(1.+dm),
                            Qp=self.QpArr[indext], Qs=self.QsArr[indext], etap=self.etapArr[indext], etas=self.etasArr[indext],
                                frefp=self.frefpArr[indext], frefs=self.frefsArr[indext], zmin=z1)
            if datatype=='qp':
                self.QpArr[index]=self.QpArr[index]*(1.+dm)
                if Hb !=0:
                    self.addlayer(H=Hb, vsv=self.VsArr[indexb], vpv=self.VpArr[indexb], rho=self.rhoArr[indexb],
                            Qp=self.QpArr[indexb]*(1.+dm), Qs=self.QsArr[indexb], etap=self.etapArr[indexb], etas=self.etasArr[indexb],
                                frefp=self.frefpArr[indexb], frefs=self.frefsArr[indexb], zmin=z2)
                if Ht !=0.:
                    self.addlayer(H=Ht, vsv=self.VsArr[indext], vpv=self.VpArr[indext], rho=self.rhoArr[indext],
                            Qp=self.QpArr[indext]*(1.+dm), Qs=self.QsArr[indext], etap=self.etapArr[indext], etas=self.etasArr[indext],
                                frefp=self.frefpArr[indext], frefs=self.frefsArr[indext], zmin=z1)
            if datatype=='qs':
                self.QsArr[index]=self.QsArr[index]*(1.+dm)
                if Hb !=0:
                    self.addlayer(H=Hb, vsv=self.VsArr[indexb], vpv=self.VpArr[indexb], rho=self.rhoArr[indexb],
                            Qp=self.QpArr[indexb], Qs=self.QsArr[indexb]*(1.+dm), etap=self.etapArr[indexb], etas=self.etasArr[indexb],
                                frefp=self.frefpArr[indexb], frefs=self.frefsArr[indexb], zmin=z2)
                if Ht !=0.:
                    self.addlayer(H=Ht, vsv=self.VsArr[indext], vpv=self.VpArr[indext], rho=self.rhoArr[indext],
                            Qp=self.QpArr[indext], Qs=self.QsArr[indext]*(1.+dm), etap=self.etapArr[indext], etas=self.etasArr[indext],
                                frefp=self.frefpArr[indext], frefs=self.frefsArr[indext], zmin=z1)
            if datatype=='etap':
                self.etapArr[index]=self.etapArr[index]*(1.+dm)
                if Hb !=0:
                    self.addlayer(H=Hb, vsv=self.VsArr[indexb], vpv=self.VpArr[indexb], rho=self.rhoArr[indexb],
                            Qp=self.QpArr[indexb], Qs=self.QsArr[indexb], etap=self.etapArr[indexb]*(1.+dm), etas=self.etasArr[indexb],
                                frefp=self.frefpArr[indexb], frefs=self.frefsArr[indexb], zmin=z2)
                if Ht !=0.:
                    self.addlayer(H=Ht, vsv=self.VsArr[indext], vpv=self.VpArr[indext], rho=self.rhoArr[indext],
                            Qp=self.QpArr[indext], Qs=self.QsArr[indext], etap=self.etapArr[indext]*(1.+dm), etas=self.etasArr[indext],
                                frefp=self.frefpArr[indext], frefs=self.frefsArr[indext], zmin=z1)
            if datatype=='etas':
                self.etasArr[index]=self.etasArr[index]*(1.+dm)
                if Hb !=0:
                    self.addlayer(H=Hb, vsv=self.VsArr[indexb], vpv=self.VpArr[indexb], rho=self.rhoArr[indexb],
                            Qp=self.QpArr[indexb], Qs=self.QsArr[indexb], etap=self.etapArr[indexb], etas=self.etasArr[indexb]*(1.+dm),
                                frefp=self.frefpArr[indexb], frefs=self.frefsArr[indexb], zmin=z2)
                if Ht !=0.:
                    self.addlayer(H=Ht, vsv=self.VsArr[indext], vpv=self.VpArr[indext], rho=self.rhoArr[indext],
                            Qp=self.QpArr[indext], Qs=self.QsArr[indext], etap=self.etapArr[indext], etas=self.etasArr[indext]*(1.+dm),
                                frefp=self.frefpArr[indext], frefs=self.frefsArr[indext], zmin=z1)
            if datatype=='frefp':
                self.frefpArr[index]=self.frefpArr[index]*(1.+dm)
                if Hb !=0:
                    self.addlayer(H=Hb, vsv=self.VsArr[indexb], vpv=self.VpArr[indexb], rho=self.rhoArr[indexb],
                            Qp=self.QpArr[indexb], Qs=self.QsArr[indexb], etap=self.etapArr[indexb], etas=self.etasArr[indexb],
                                frefp=self.frefpArr[indexb]*(1.+dm), frefs=self.frefsArr[indexb], zmin=z2)
                if Ht !=0.:
                    self.addlayer(H=Ht, vsv=self.VsArr[indext], vpv=self.VpArr[indext], rho=self.rhoArr[indext],
                            Qp=self.QpArr[indext], Qs=self.QsArr[indext], etap=self.etapArr[indext], etas=self.etasArr[indext],
                                frefp=self.frefpArr[indext]*(1.+dm), frefs=self.frefsArr[indext], zmin=z1)
            if datatype=='frefs':
                self.frefsArr[index]=self.frefsArr[index]*(1.+dm)
                if Hb !=0:
                    self.addlayer(H=Hb, vsv=self.VsArr[indexb], vpv=self.VpArr[indexb], rho=self.rhoArr[indexb],
                            Qp=self.QpArr[indexb], Qs=self.QsArr[indexb], etap=self.etapArr[indexb], etas=self.etasArr[indexb],
                                frefp=self.frefpArr[indexb], frefs=self.frefsArr[indexb]*(1.+dm), zmin=z2)
                if Ht !=0.:
                    self.addlayer(H=Ht, vsv=self.VsArr[indext], vpv=self.VpArr[indext], rho=self.rhoArr[indext],
                            Qp=self.QpArr[indext], Qs=self.QsArr[indext], etap=self.etapArr[indext], etas=self.etasArr[indext],
                                frefp=self.frefpArr[indext], frefs=self.frefsArr[indext]*(1.+dm), zmin=z1)
        elif self.modeltype == 'TRANSVERSE ISOTROPIC':
            if datatype=='rho':
                self.rhoArr[index]=self.rhoArr[index]*(1.+dm)
                if Hb != 0:
                    self.addlayer(H=Hb, vsv=self.VsvArr[indexb], vpv=self.VpvArr[indexb], vsh=self.VshArr[indexb],
                        vph=self.VphArr[indexb], vpf=self.VpfArr[indexb], rho=self.rhoArr[indexb]*(1.+dm),
                            Qp=self.QpArr[indexb], Qs=self.QsArr[indexb], etap=self.etapArr[indexb], etas=self.etasArr[indexb],
                                frefp=self.frefpArr[indexb], frefs=self.frefsArr[indexb], zmin=z2)
                if Ht != 0.:
                    self.addlayer(H=Ht, vsv=self.VsvArr[indext], vpv=self.VpvArr[indext], vsh=self.VshArr[indext],
                        vph=self.VphArr[indext], vpf=self.VpfArr[indext], rho=self.rhoArr[indext]*(1.+dm),
                            Qp=self.QpArr[indext], Qs=self.QsArr[indext], etap=self.etapArr[indext], etas=self.etasArr[indext],
                                frefp=self.frefpArr[indext], frefs=self.frefsArr[indext], zmin=z1)
            if datatype=='qp':
                self.QpArr[index]=self.QpArr[index]*(1.+dm)
                if Hb != 0:
                    self.addlayer(H=Hb, vsv=self.VsvArr[indexb], vpv=self.VpvArr[indexb], vsh=self.VshArr[indexb],
                        vph=self.VphArr[indexb], vpf=self.VpfArr[indexb], rho=self.rhoArr[indexb],
                            Qp=self.QpArr[indexb]*(1.+dm), Qs=self.QsArr[indexb], etap=self.etapArr[indexb], etas=self.etasArr[indexb],
                                frefp=self.frefpArr[indexb], frefs=self.frefsArr[indexb], zmin=z2)
                if Ht != 0.:
                    self.addlayer(H=Ht, vsv=self.VsvArr[indext], vpv=self.VpvArr[indext], vsh=self.VshArr[indext],
                        vph=self.VphArr[indext], vpf=self.VpfArr[indext], rho=self.rhoArr[indext],
                            Qp=self.QpArr[indext]*(1.+dm), Qs=self.QsArr[indext], etap=self.etapArr[indext], etas=self.etasArr[indext],
                                frefp=self.frefpArr[indext], frefs=self.frefsArr[indext], zmin=z1)
            if datatype=='qs':
                self.QsArr[index]=self.QsArr[index]*(1.+dm)
                if Hb != 0:
                    self.addlayer(H=Hb, vsv=self.VsvArr[indexb], vpv=self.VpvArr[indexb], vsh=self.VshArr[indexb],
                        vph=self.VphArr[indexb], vpf=self.VpfArr[indexb], rho=self.rhoArr[indexb],
                            Qp=self.QpArr[indexb], Qs=self.QsArr[indexb]*(1.+dm), etap=self.etapArr[indexb], etas=self.etasArr[indexb],
                                frefp=self.frefpArr[indexb], frefs=self.frefsArr[indexb], zmin=z2)
                if Ht != 0.:
                    self.addlayer(H=Ht, vsv=self.VsvArr[indext], vpv=self.VpvArr[indext], vsh=self.VshArr[indext],
                        vph=self.VphArr[indext], vpf=self.VpfArr[indext], rho=self.rhoArr[indext],
                            Qp=self.QpArr[indext], Qs=self.QsArr[indext]*(1.+dm), etap=self.etapArr[indext], etas=self.etasArr[indext],
                                frefp=self.frefpArr[indext], frefs=self.frefsArr[indext], zmin=z1)
            if datatype=='etap':
                self.etapArr[index]=self.etapArr[index]*(1.+dm)
                if Hb != 0:
                    self.addlayer(H=Hb, vsv=self.VsvArr[indexb], vpv=self.VpvArr[indexb], vsh=self.VshArr[indexb],
                        vph=self.VphArr[indexb], vpf=self.VpfArr[indexb], rho=self.rhoArr[indexb],
                            Qp=self.QpArr[indexb], Qs=self.QsArr[indexb], etap=self.etapArr[indexb]*(1.+dm), etas=self.etasArr[indexb],
                                frefp=self.frefpArr[indexb], frefs=self.frefsArr[indexb], zmin=z2)
                if Ht != 0.:
                    self.addlayer(H=Ht, vsv=self.VsvArr[indext], vpv=self.VpvArr[indext], vsh=self.VshArr[indext],
                        vph=self.VphArr[indext], vpf=self.VpfArr[indext], rho=self.rhoArr[indext],
                            Qp=self.QpArr[indext], Qs=self.QsArr[indext], etap=self.etapArr[indext]*(1.+dm), etas=self.etasArr[indext],
                                frefp=self.frefpArr[indext], frefs=self.frefsArr[indext], zmin=z1)
            if datatype=='etas':
                self.etasArr[index]=self.etasArr[index]*(1.+dm)
                if Hb != 0:
                    self.addlayer(H=Hb, vsv=self.VsvArr[indexb], vpv=self.VpvArr[indexb], vsh=self.VshArr[indexb],
                        vph=self.VphArr[indexb], vpf=self.VpfArr[indexb], rho=self.rhoArr[indexb],
                            Qp=self.QpArr[indexb], Qs=self.QsArr[indexb], etap=self.etapArr[indexb], etas=self.etasArr[indexb]*(1.+dm),
                                frefp=self.frefpArr[indexb], frefs=self.frefsArr[indexb], zmin=z2)
                if Ht != 0.:
                    self.addlayer(H=Ht, vsv=self.VsvArr[indext], vpv=self.VpvArr[indext], vsh=self.VshArr[indext],
                        vph=self.VphArr[indext], vpf=self.VpfArr[indext], rho=self.rhoArr[indext],
                            Qp=self.QpArr[indext], Qs=self.QsArr[indext], etap=self.etapArr[indext], etas=self.etasArr[indext]*(1.+dm),
                                frefp=self.frefpArr[indext], frefs=self.frefsArr[indext], zmin=z1)
            if datatype=='frefp':
                self.frefpArr[index]=self.frefpArr[index]*(1.+dm)
                if Hb != 0:
                    self.addlayer(H=Hb, vsv=self.VsvArr[indexb], vpv=self.VpvArr[indexb], vsh=self.VshArr[indexb],
                        vph=self.VphArr[indexb], vpf=self.VpfArr[indexb], rho=self.rhoArr[indexb],
                            Qp=self.QpArr[indexb], Qs=self.QsArr[indexb], etap=self.etapArr[indexb], etas=self.etasArr[indexb],
                                frefp=self.frefpArr[indexb]*(1.+dm), frefs=self.frefsArr[indexb], zmin=z2)
                if Ht != 0.:
                    self.addlayer(H=Ht, vsv=self.VsvArr[indext], vpv=self.VpvArr[indext], vsh=self.VshArr[indext],
                        vph=self.VphArr[indext], vpf=self.VpfArr[indext], rho=self.rhoArr[indext],
                            Qp=self.QpArr[indext], Qs=self.QsArr[indext], etap=self.etapArr[indext], etas=self.etasArr[indext],
                                frefp=self.frefpArr[indext]*(1.+dm), frefs=self.frefsArr[indext], zmin=z1)
            if datatype=='frefs':
                self.frefsArr[index]=self.frefsArr[index]*(1.+dm)
                if Hb != 0:
                    self.addlayer(H=Hb, vsv=self.VsvArr[indexb], vpv=self.VpvArr[indexb], vsh=self.VshArr[indexb],
                        vph=self.VphArr[indexb], vpf=self.VpfArr[indexb], rho=self.rhoArr[indexb],
                            Qp=self.QpArr[indexb], Qs=self.QsArr[indexb], etap=self.etapArr[indexb], etas=self.etasArr[indexb],
                                frefp=self.frefpArr[indexb], frefs=self.frefsArr[indexb]*(1.+dm), zmin=z2)
                if Ht != 0.:
                    self.addlayer(H=Ht, vsv=self.VsvArr[indext], vpv=self.VpvArr[indext], vsh=self.VshArr[indext],
                        vph=self.VphArr[indext], vpf=self.VpfArr[indext], rho=self.rhoArr[indext],
                            Qp=self.QpArr[indext], Qs=self.QsArr[indext], etap=self.etapArr[indext], etas=self.etasArr[indext],
                                frefp=self.frefpArr[indext], frefs=self.frefsArr[indext]*(1.+dm), zmin=z1)
        return
    
    def read_axisem_bm(self, infname):
        """
        Read 1D block model from AxiSEM
        """
        self.earthtype  = 'SPHERICAL EARTH'
        with open(infname, 'rb') as f:
            f.readline()
            cline           = f.readline()
            cline           = cline.split()
            if cline[0] != 'NAME':
                raise ValueError('Unexpected header: '+cline[0])
            self.modelname  = cline[1]
            f.readline()
            cline           = f.readline()
            cline           = cline.split()
            if cline[0] != 'ANISOTROPIC':
                raise ValueError('Unexpected header: '+cline[0])
            anisotropic     = cline[1]
            if anisotropic == 'T':
                self.modeltype  = 'TRANSVERSE ISOTROPIC'
                self.modelheader= '\tH(KM)\tVPV(KM/S)\tVSV(KM/S)\tRHO(GM/CC)\tQP\tQS\tETAP\tETAS\tFREFP\tFREFS\n\tVPH(KM/S)\tVSH(KM/S)\tVPF(KM/S)'
                self.VsvArr     = np.array([])
                self.VpvArr     = np.array([])
                self.VshArr     = np.array([])
                self.VphArr     = np.array([])
                self.VpfArr     = np.array([])
            elif anisotropic == 'F':
                self.modeltype  = 'ISOTROPIC'
                self.modelheader= '\tH(KM)\tVP(KM/S)\tVS(KM/S)\tRHO(GM/CC)\tQP\tQS\tETAP\tETAS\tFREFP\tFREFS'
                self.VsArr      = np.array([])
                self.VpArr      = np.array([])
            cline           = f.readline()
            cline           = cline.split()
            if cline[0] != 'UNITS':
                raise ValueError('Unexpected header: '+cline[0])
            if cline[1] == 'm': unit = 1000.
            elif cline[1] == 'km': unit = 1.
            cline           = f.readline()
            cline           = cline.split()
            if cline[0] != 'COLUMNS':
                raise ValueError('Unexpected header: '+cline[0])
            ind = {}
            i=0
            for hdrstr in cline[1:]:
                ind[hdrstr] = i
                i   += 1
            ###
            # Read model parameters
            ###
            z0 = 0.
            for line in f.readlines():
                cline   = line.split()
                if cline[0] == '#': continue
                r   = float(cline[ ind['radius'] ])/unit
                z   = 6371. - r
                H   = z - z0
                # # # print ' '.join(cline), H
                if H == 0.:
                    vpvt = float(cline[ ind['vpv'] ])/unit
                    vsvt = float(cline[ ind['vsv'] ])/unit
                    rhot = float(cline[ ind['rho'] ])/unit
                    qkat = float(cline[ ind['qka'] ])
                    qmut = float(cline[ ind['qmu'] ])
                    if qmut != 0.:
                        qpt  = qmut/ ( 4.*(vsvt/vpvt)**2/3. + (1.- 4.*(vsvt/vpvt)**2/3.) * qmut/qkat )
                    else:
                        qpt  = 57822.
                    if anisotropic == 'T':
                        vpht = float(cline[ ind['vph'] ])/unit
                        vsht = float(cline[ ind['vsh'] ])/unit
                        etat = float(cline[ ind['eta'] ])/unit
                        vpft = np.sqrt(etat*(vpht**2 - vsvt**2))
                    # # # print vsvt
                    continue
                vpvb= float(cline[ ind['vpv'] ])/unit
                vsvb= float(cline[ ind['vsv'] ])/unit
                rhob= float(cline[ ind['rho'] ])/unit
                qkab= float(cline[ ind['qka'] ])
                qmub= float(cline[ ind['qmu'] ])
                if qmub != 0.:
                    qpb  = qmub/ ( 4.*(vsvb/vpvb)**2/3. + (1.- 4.*(vsvb/vpvb)**2/3.) * qmub/qkab )
                else:
                    qpb  = 57822.
                if anisotropic == 'T':
                    vphb = float(cline[ ind['vph'] ])/unit
                    vshb = float(cline[ ind['vsh'] ])/unit
                    etab = float(cline[ ind['eta'] ])/unit
                    vpfb = np.sqrt(etab*(vphb**2 - vsvb**2))
                self.HArr   = np.append(self.HArr, H)
                self.rhoArr = np.append(self.rhoArr, (rhot+rhob)/2.)
                self.QpArr  = np.append(self.QpArr, (qpt+qpb)/2.)
                self.QsArr  = np.append(self.QsArr, (qmut+qmub)/2.)
                if anisotropic == 'T':
                    self.VsvArr = np.append(self.VsvArr, (vsvt+vsvb)/2.)
                    self.VshArr = np.append(self.VshArr, (vsht+vshb)/2.)
                    self.VpvArr = np.append(self.VpvArr, (vpvt+vpvb)/2.)
                    self.VphArr = np.append(self.VphArr, (vpht+vphb)/2.)
                    self.VpfArr = np.append(self.VpfArr, (vpft+vpfb)/2.)
                else:
                    self.VsArr = np.append(self.VsArr, (vsvt+vsvb)/2.)
                    self.VpArr = np.append(self.VpArr, (vpvt+vpvb)/2.)
                # # # print vsvt, vsvb, (vsvt+vsvb)/2.
                z0      = z
                vpvt    = vpvb
                vsvt    = vsvb
                rhot    = rhob
                qkat    = qkab
                qmut    = qmub
                qpt     = qpb
                if anisotropic == 'T':
                    vpht = vphb
                    vsht = vshb
                    etat = etab
                    vpft = vpfb
        self.etapArr    = np.zeros(self.HArr.size)
        self.etasArr    = np.zeros(self.HArr.size)
        self.frefpArr   = np.ones(self.HArr.size)
        self.frefsArr   = np.ones(self.HArr.size)
        self.DepthArr   = np.cumsum(self.HArr)
        return
    
    def write_axisem_bm(self, outfname, noq=True, modelname='model_cps', comment='Model from CPS', unit=1000.):
        with open(outfname, 'wb') as f:
            ###
            # header
            ###
            f.writelines('# %s\n' %comment)
            f.writelines('NAME\t%s\n' %modelname)
            if noq: f.writelines('ANELASTIC\tF\n')
            else: f.writelines('ANELASTIC\tT\n')
            if self.modeltype == 'ISOTROPIC': f.writelines('ANISOTROPIC\tF\n')
            else: f.writelines('ANISOTROPIC\tT\n')
            if unit == 1000.: f.writelines('UNITS\tm\n')
            elif unit == 1.: f.writelines('UNITS\tkm\n')
            else: raise ValueError('Unexpected units!')
            if self.modeltype == 'ISOTROPIC' and noq:
                f.writelines('COLUMNS\tradius\trho\tvpv\tvsv\n')
            elif self.modeltype == 'ISOTROPIC' and not noq:
                f.writelines('COLUMNS\tradius\trho\tvpv\tvsv\tqka\tqmu\n')
            elif self.modeltype == 'TRANSVERSE ISOTROPIC' and noq:
                f.writelines('COLUMNS\tradius\trho\tvpv\tvsv\tvph\tvsh\teta\n')
            else:
                f.writelines('COLUMNS\tradius\trho\tvpv\tvsv\tqka\tqmu\tvph\tvsh\teta\n')
            ###
            # model parameters
            ###
            N   = self.HArr.size
            topArr  = self.DepthArr - self.HArr
            for i in xrange(N):
                z0  = topArr[i]; z1 = self.DepthArr[i]
                r0  = (6371.-z0)*unit; r1 = (6371.-z1)*unit
                rho = self.rhoArr[i]*unit
                qka = self.QpArr[i]
                qmu = self.QsArr[i]
                if self.modeltype == 'ISOTROPIC' and noq:
                    vpv = self.VpArr[i]*unit
                    vsv = self.VsArr[i]*unit
                    f.writelines('%g\t%g\t%g\t%g\n' %(r0, rho, vpv, vsv))
                    f.writelines('%g\t%g\t%g\t%g\n' %(r1, rho, vpv, vsv))
                elif self.modeltype == 'ISOTROPIC' and not noq:
                    vpv = self.VpArr[i]*unit
                    vsv = self.VsArr[i]*unit
                    f.writelines('%g\t%g\t%g\t%g\t%g\t%g\n' %(r0, rho, vpv, vsv, qka, qmu))
                    f.writelines('%g\t%g\t%g\t%g\t%g\t%g\n' %(r1, rho, vpv, vsv, qka, qmu))
                elif self.modeltype == 'TRANSVERSE ISOTROPIC' and noq:
                    vpv = self.VpvArr[i]*unit
                    vsv = self.VsvArr[i]*unit
                    vph = self.VphArr[i]*unit
                    vsh = self.VshArr[i]*unit
                    vpf = self.VpfArr[i]*unit
                    eta = vpf**2/(vph**2 - 2.*vsv**2)
                    f.writelines('\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' %(r0, rho, vpv, vsv, vph, vsh, eta))
                    f.writelines('\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' %(r1, rho, vpv, vsv, vph, vsh, eta))
                else:
                    vpv = self.VpvArr[i]*unit
                    vsv = self.VsvArr[i]*unit
                    vph = self.VphArr[i]*unit
                    vsh = self.VshArr[i]*unit
                    vpf = self.VpfArr[i]*unit
                    eta = vpf**2/(vph**2 - 2.*vsv**2)
                    f.writelines('\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' %(r0, rho, vpv, vsv, qka, qmu, vph, vsh, eta))
                    f.writelines('\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' %(r1, rho, vpv, vsv, qka, qmu, vph, vsh, eta))
        
                    

    
class vprofile(object):
    def __init__(self, vsArr=np.array([]), vpArr=np.array([]), rhoArr=np.array([]), RmaxArr=np.array([]), RminArr=np.array([]),
                 z0Arr=np.array([]), HArr=np.array([]), xArr=np.array([]), yArr=np.array([]), dtypeArr=np.array([]), infname=None ):
        """
        An object to handle vertical profile, will be used by CPSPy
        This object is only used for 2D/3D comparison of SPECFEM2D and SW4
        ========================================================================================
        output txt format:
        vs/dvs    vp/dvp    rho/drho    Rmax    Rmin    z0    H    x    y    dtype
        
        dtype:
        0. absolute value for vs/vp/rho
        1. percentage value for dvs/dvp/drho 
        ========================================================================================
        """
        self.vsArr      = vsArr
        self.vpArr      = vpArr
        self.rhoArr     = rhoArr
        self.RmaxArr    = RmaxArr
        self.RminArr    = RminArr
        self.z0Arr      = z0Arr
        self.HArr       = HArr
        self.xArr       = xArr
        self.yArr       = yArr
        self.dtypeArr   = dtypeArr
        if infname != None:
            self.read(infname=infname)
        return
    
    def read(self, infname):
        inArr=np.loadtxt(infname)
        try:
            self.vsArr      = inArr[:,0]
            self.vpArr      = inArr[:,1]
            self.rhoArr     = inArr[:,2]
            self.RmaxArr    = inArr[:,3]
            self.RminArr    = inArr[:,4]
            self.z0Arr      = inArr[:,5]
            self.HArr       = inArr[:,6]
            self.xArr       = inArr[:,7]
            self.yArr       = inArr[:,8]
            self.dtypeArr   = inArr[:,9]
        except:
            self.vsArr      = np.array([inArr[0]])
            self.vpArr      = np.array([inArr[1]])
            self.rhoArr     = np.array([inArr[2]])
            self.RmaxArr    = np.array([inArr[3]])
            self.RminArr    = np.array([inArr[4]])
            self.z0Arr      = np.array([inArr[5]])
            self.HArr       = np.array([inArr[6]])
            self.xArr       = np.array([inArr[7]])
            self.yArr       = np.array([inArr[8]])
            self.dtypeArr   = np.array([inArr[9]])
        return
    
    def write(self, outfname):
        N       = self.vsArr.size
        outArr  = np.append(self.vsArr, self.vpArr)
        outArr  = np.append(outArr, self.rhoArr)
        outArr  = np.append(outArr, self.RmaxArr)
        outArr  = np.append(outArr, self.RminArr)
        outArr  = np.append(outArr, self.z0Arr)
        outArr  = np.append(outArr, self.HArr)
        outArr  = np.append(outArr, self.xArr)
        outArr  = np.append(outArr, self.yArr)
        outArr  = np.append(outArr, self.dtypeArr)
        outArr  = outArr.reshape(10, N)
        outArr  = outArr.T
        self.check()
        np.savetxt(outfname, outArr, fmt='%g', header='vs/dvs vp/dvp rho/drho Rmax Rmin z0 H x y dtype(0: m, 1: dm)')
        return
    
    def check(self):
        N = self.vsArr.size
        for i in xrange(N):
            x       = self.xArr[i]
            y       = self.yArr[i]
            Rmax    = self.RmaxArr[i]
            H       = self.HArr[i]
            index   = (self.xArr==x)* (self.yArr==y) * (self.RmaxArr==Rmax) * (self.HArr==H)
            if np.any(index):
                warnings.warn('Profile of same geometry and location exists.', UserWarning, stacklevel=1)
                return
        return
    
    def add(self, Rmax, Rmin, z0, H, x, y, dtype, vs=None, vp=None, rho=None):
        """
        Two kinds of input:
        1. Rmax or x is an array, the function will assume N vertical profile input(N=Rmax.size or x.size)
        2. If not, the function will first check if there is any previous profile with same x, y, Rmax. If yes, change
            vs, vp or rho in the existing profile; if not, add a new profile
        """
        if isinstance(x, np.ndarray) or isinstance(Rmax, np.ndarray):
            if vs ==None or vp == None or rho == None:
                raise ValueError('For array input, all model parameters must be assigned !')
            try:
                N = x.size
            except:
                N = Rmax.size
            ##################################################
            # Making all the input variables to be numpy array of size N
            ##################################################
            if not isinstance(Rmax, np.ndarray):
                Rmax    = np.ones(N)*Rmax
            if not isinstance(Rmin, np.ndarray):
                Rmin    = np.ones(N)*Rmin
            if not isinstance(z0, np.ndarray):
                z0      = np.ones(N)*z0
            if not isinstance(H, np.ndarray):
                H       = np.ones(N)*H
            if not isinstance(x, np.ndarray):
                x       = np.ones(N)*x
            if not isinstance(y, np.ndarray):
                y       = np.ones(N)*y
            if not isinstance(dtype, np.ndarray):
                dtype   = np.ones(N)*dtype
            if not isinstance(vs, np.ndarray):
                vs      = np.ones(N)*vs
            if not isinstance(vp, np.ndarray):
                vp      = np.ones(N)*vp
            if not isinstance(rho, np.ndarray):
                rho     = np.ones(N)*rho
        else:
            index       = (self.xArr==x)* (self.yArr==y) * (self.RmaxArr==Rmax) * (self.HArr==H)
            # change model parameters if profile with same geometry/location exist
            if np.any(index):
                if vs !=None:
                    self.vsArr[index]   = vs
                if vp !=None:
                    self.vpArr[index]   = vp
                if rho !=None:
                    self.rhoArr[index]  = rho
                return
            if dtype==0 and (vs ==None or vp == None or rho == None):
                raise ValueError('For absolute value profile, all model parameters must be assigned !')
            if vs==None:
                vs  = 0.
            if vp == None:
                vp  = 0.
            if rho == None:
                rho = 0.
        self.vsArr      = np.append( self.vsArr, vs)
        self.vpArr      = np.append( self.vpArr, vp)
        self.rhoArr     = np.append( self.rhoArr, rho)
        self.xArr       = np.append( self.xArr, x)
        self.yArr       = np.append( self.yArr, y)
        self.RmaxArr    = np.append( self.RmaxArr, Rmax)
        self.RminArr    = np.append( self.RminArr, Rmin)
        self.HArr       = np.append( self.HArr, H)
        self.z0Arr      = np.append( self.z0Arr, z0)
        self.dtypeArr   = np.append( self.dtypeArr, dtype)
        return
    

        
    