# (c) Hilty Lab, 2022-2023

import numpy as np
import json
import os

from .expdta import timedata, freqdata, integraldata
from .pulsebase import pulsebase

class expbase():
    def __init__(self, p={}, pproc={}, pinc={}, pstat={}, ppre={}):

        self.pstat=pstat
        self.p=p
        self.pproc=pproc
        self.pinc=pinc
        self.ppre=ppre

        sr=self.pproc.get('srate',1)
        self.tdt=[timedata([],1/sr)]   # time data, frequency data are re-generated automatically at every parameter change
        self.frq=[freqdata()]

        # integral data (self.idt), 2d data should be generated on the fly when needed

    # get phase parameter value
    def getphparm(self):
        ph=self.pproc.get('ph',None)
        if not (type(ph)==int or type(ph)==float):
            ph=None
        return(ph)

    # get sample rate
    def getsrate(self):
        srate=self.pstat.get('sratex',None)  # use actual sample rate if available
        if srate is None:
            srate=self.p['srate']
        return(srate)

    # process data and regenerate integrals
    def proc(self):
        ph=self.getphparm()
        x1=self.pproc.get('ftmin',0)
        x2=self.pproc.get('ftmax',0)
        wdw=self.pproc.get('wdw','none')
        wdwr=self.pproc.get('wdwr',1)
        fc=self.pproc.get('fc',0)

        self.frq=[]
        for td in self.tdt:
            frq=td.ft(x1,x2,ph,wdw,wdwr,fc)
            c=frq.x_to_ind(self.pproc.get('ffmin',0))
            d=frq.x_to_ind(self.pproc.get('ffmax',0))
            frq.cut(c,d)
            if(self.pproc['ftdisp']=='log'):
                frq.to_log()
            self.frq.append(frq)

    # average multiple scans
    def avgmulti(self):
        nscan=self.nscan()
        if(nscan>1):
            td=self.tdt[0]
            for i in range(nscan-1):
                td.dta=td.dta+self.tdt[i+1].dta
            self.tdt=[timedata(td.dta/nscan,td.dx)]
            self.proc()

    # print parameters
    def printparam(self,par):
        s=json.dumps(par, indent=6)
        return(s)

    # calculate SNR of spectrum
    # index [index of spectrum]
    def snr(self,index):
        if index>0:
            ph=self.getphparm()
            s1=self.pproc.get('s1',0)
            s2=self.pproc.get('s2',0)
            n1=self.pproc.get('n1',0)
            n2=self.pproc.get('n2',0)
            sind=self.pproc.get('sind',0)
            s=self.frq[index-1].calsnr(s1,s2,n1,n2,sind,ph)
        else:
            s=0
        return(s)

    def integrate(self):
        ss1=len(self.frq)
        ss2=self.frq[0].nchann()
        a=np.zeros((ss1,ss2))
        for ii,f in enumerate(self.frq):
            a2=f.integrate(self.pproc['intmin'],self.pproc['intmax'])
            a[ii]=a2
        self.idt=integraldata(a.transpose())

    # split multi-pulse experiment
    def split(self):
        ioff=None
        if(self.pproc.get('autosplit','false').lower()=='true'):  # automatic if supported by pulse program
            if self.p['ppg']=='cpmg':
                k=pulsebase(self.p)
                ioff,ilen=k.cpmg_sp()
        if ioff is None:
            ioff=self.tdt[0].x_to_ind(self.pproc['soff'])
            ilen=self.tdt[0].x_to_ind(self.pproc['slen'])
        srate=self.getsrate()
        self.pstat['ioff']=ioff
        self.pstat['ilen']=ilen
        n=(self.tdt[0].len()-ioff)//ilen
        tdt2=[]
        for i in range(n):
            lt=ioff+i*ilen
            rt=ioff+(i+1)*ilen
            tdt2.append(timedata(self.tdt[0].dta[:,lt:rt],1/srate))
        self.tdt=tdt2
        self.pstat['nsampx']=self.tdt[0].len()
        self.proc()

    # blank fid
    def blank(self,i=0):
        k=pulsebase(self.p)
        l=k.len()
        n=round(self.pproc['blankb']/k.dx)
        f=np.convolve(np.ones(n),k.dta[0]!=0)==0
        self.tdt[i].dta=np.multiply(self.tdt[i].dta,f[0:l])

    # apply digital filter
    def digfilt(self):
        for t in self.tdt:
            digfmin=self.pproc['digfmin']
            digfmax=self.pproc['digfmax']
            t.butter([digfmin,digfmax])
            self.pstat['digfmin']=digfmin
            self.pstat['digfmax']=digfmax

    # save parameters and scan in memory
    def save(self,filename):
        r,_=os.path.splitext(filename)
        filename0=r+'.bin'
        filename1=r+'.json'
        if os.path.exists(filename0):
            os.remove(filename0)
        with open(filename0,"ab") as f:
            for td in self.tdt:
                td.save(f)
        self.saveparams(filename1)

    # save the parameter file
    def saveparams(self,filename):
        with open(filename,"w") as outfile:
            json.dump((self.p,self.pproc,self.pinc,self.ppre,self.pstat),outfile,indent=6)

    # number of scans in the experiment
    def nscan(self):
        return(len(self.tdt))

    # list of traces to display
    def getdisplist(self):
        d=self.pproc.get('disp',None)
        if d=="none":
            d=None
        return(d)

    # load parameters and single scan
    def load(self,filename):
        r,_=os.path.splitext(filename)
        filename0=r+'.bin'
        filename1=r+'.json'
        with open(filename1,"r") as infile:
            a=json.load(infile)
        (self.p,self.pproc,self.pinc,self.ppre,self.pstat)=a
        dta=np.fromfile(filename0,dtype='float64')
        nchann=self.pstat['ninchann']
        srate=self.getsrate()
        nsamp=self.pstat.get('nsampx',None)  # use actual number of samples
        if nsamp is None:
            nsamp=self.p['nsamp']
            skip=self.p.get('nskip',0)
            nsamp=nsamp-skip
        nscan=np.size(dta)//nchann//nsamp
        self.tdt=[]
        lgt=nchann*nsamp
        if(nscan>0):
            for i in range(nscan):
                dt=dta[i*lgt:(i+1)*lgt].reshape([nchann,nsamp])
                self.tdt.append(timedata(dt,1/srate))
        else:
            self.tdt=[timedata([],1/srate)]
        self.proc()

    # plot time graph with predefined axes. index=0 plots all
    def plottm(self,ax,index):
        d=self.getdisplist()
        ax.clear()
        if index>0:
            self.tdt[index-1].plot(ax,disp=d)
        else:
            arr=timedata(self.tdt[0].dta,self.tdt[0].dx)
            nscan=self.nscan()
            if(nscan>1):
                for i in range(nscan-1):
                    arr.append(self.tdt[i+1].dta)
            arr.plot(ax,"delta t [s]",d)

    # export audio file
    def exportaudio(self,filename,index,trace=0):
        if index>0:
            self.tdt[index-1].exportaudio(filename,trace)
        else:
            arr=timedata(self.tdt[0].dta,self.tdt[0].dx)
            nscan=self.nscan()
            if(nscan>1):
                for i in range(nscan-1):
                    arr.append(self.tdt[i+1].dta)
            arr.exportaudio(filename,trace)

    # plot frequency graph with predefined axes. index=0 plots all
    def plotfrq(self,ax,index):
        d=self.getdisplist()
        ax.clear()
        if index>0:
            self.frq[index-1].plot(ax,disp=d)
        else:
            arr=freqdata(self.frq[0].dta,self.frq[0].dx)
            nscan=self.nscan()
            if(nscan>1):
                for i in range(nscan-1):
                    arr.append(self.frq[i+1].dta)

            dispper=self.pproc.get('dispper',0)     # omitted percentage at the beginning and end of each concatenated slice
            if dispper==0:                          # inactive if not specified or set to 0
                pass
            else:
                seg=np.ones_like(self.frq[0].dta)
                l=len(self.frq[0].dta[0])
                k=int(dispper*l/2)
                seg[:,:k]=np.nan
                seg[:,l-k:]=np.nan
                mask=np.array(seg)
                i=1
                while i < nscan:
                    mask=np.hstack((mask,seg))      # creating mask function consisted of 1 and np.nan
                    i+=1
                arr.dta=np.multiply(arr.dta,mask)

            arr.plot(ax,"delta f [Hz]",d)
     
    #### Functions for 2d spectra        
    # process 2d spectrum        
    def proc2(self):
        if len(self.tdt)>1:
            #process first dimension
            self.proc()
            
            n2=self.len2()//2 #number of indirect complex points
            n1=self.frq[0].len()
            mode=self.pproc.get('mode','complex')
            
            #indirect dimension complex array
            self.r2d=np.zeros([n2,n1],dtype=complex)
            i=0
            for i in range(n2):
                if mode=='sttp':
                    fc=(-1)**(i%2)
                else:
                    fc=1
                self.r2d[i]=fc*(self.frq[2*i].dta-1j*self.frq[2*i+1].dta)
                i=i+1
            
            #fourier transform indirect dimension
            self.s2d=np.fft.fftshift(np.fft.fft(self.r2d,axis=0),axes=0)
            #self.s=self.s*np.exp(1j*ph)

    # plot 2d spectrum with predefined axes      
    def plot2d(self,ax):
        try:
            #axes
            x0=self.frq[0].xvect()
            y0=self.xvect2()

            #plot
            x,y=np.meshgrid(x0,y0)
            pl=np.log(self.s2d).real
            mv=np.max(pl)
            vm=np.min(pl)+10*np.std(pl)
            
            ax.clear()
            pl=ax.pcolor(x,y,pl,vmin=vm,vmax=mv)
            ax.set_xlabel('f2 [Hz]')
            ax.set_ylabel('f1 [Hz]')
            pl.figure.canvas.draw()
            return(pl)
        except:
            ax.clear()
            ax.set_xlabel('f2 [Hz]')
            ax.set_ylabel('f1 [Hz]')
            ax.figure.canvas.draw()
            return(False)
        
    # find number of real data points in indirect dimension
    def len2(self):
        n2=len(self.frq)
        return(n2)

    # find time increment in indirect dimension
    def dw2(self):
        mode=self.pproc.get('mode','complex')
        if mode=='sttp':
            dw=self.p['dt1'] #the states/tppi pulse program uses dt1 as indirect dimension increment
        else:
            dw=self.pinc['inc'][0] #in the 1D version of the pulse program, the indirect dimension is incremented with pinc        
        return(dw)
    
    # find vector of indirect dimension frequency values
    def xvect2(self):
        dw=self.dw2()
        n2=self.len2()//2
        y0=np.linspace(-1/dw/2,1/dw/2,n2,endpoint=False)
        return(y0)
    
    # find index of indirect dimension data point with independent axis value of x
    # ensures no indices are out of bounds        
    def x_to_ind2(self, x):
        dw=self.dw2()
        x0=-1/dw/2
        n2=self.len2()//2
        dx=dw/n2
        i=round((x-self.x0)/self.dx)
        if(i<0):
            i=0
        if(i>n2-1):
            i=n2-1        
        return(i)
        
    # find value of indirect dimension axis at index
    def ind_to_x2(self,i):
        dw=self.dw2()
        x0=-1/dw/2
        x=i*dw+x0
        return(x)

    #### functions for analysis
    # average multiple spectra
    def avgmultispec(self):
        nscan=self.nscan()
        if(nscan>1):
            td=self.frq[0]
            for i in range(nscan-1):
                td.dta=td.dta+self.frq[i+1].dta
            self.frq=[freqdata(td.dta/nscan,td.dx)]

