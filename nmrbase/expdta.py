# (c) Hilty Lab, 2022-2023

import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, filtfilt

# base class for arrayed data
# contains numpy 2D array [i,k]; i: data trace index, k: data point index
# removes nskip data points from beginning of trace
class data():
    def __init__(self, dta=[], dx=1, x0=0,nskip=0):
        dd=np.array(dta)
        if np.size(np.shape(dd))==1:
            self.dta=np.array([dd])
        else:
            self.dta=dd
        self.dta=self.dta[:,nskip:]
        self.dx=dx
        self.x0=x0

    # index bounds for x1 and x2
    # entire range if second index is not larger than first
    def ibounds(self,x1=0,x2=0):
        a=self.x_to_ind(x1)
        b=self.x_to_ind(x2)
        if not(a<b):
            a=0
            b=self.len()
        return(a,b)

    # index bounds from center
    # fc [fraction of array]
    def fcbounds(self,fc):
        ln=self.len()
        if fc<=0:
            a=0
            b=ln
        else:           
            ln=self.len()
            m=round((1-fc)*ln/2)
            a=m
            b=ln-m
        return(a,b)

    # length of data array
    def len(self):
        return(np.shape(self.dta)[1])

    #number of data channels
    def nchann(self):
        return(np.shape(self.dta)[0])

    # maximum independent axis value
    def xmax(self):
        return((np.shape(self.dta)[1]-1)*self.dx)
    
    # vector containing x-axis
    def xvect(self):
        a=self.len()
        xv=np.linspace(0,a*self.dx,a,endpoint=False)+self.x0
        return(xv)

    # find index of data point with independent axis value of x
    # ensures no indices are out of bounds
    def x_to_ind(self,x,bounds=True):
        i=round((x-self.x0)/self.dx)
        if(bounds):
            if(i<0):
                i=0
            if(i>self.len()-1):
                i=max(0,self.len()-1)
        return(i)

    # find value of independent axis at index
    def ind_to_x(self,i):
        x=i*self.dx+self.x0
        return(x)

    # cut a slice between indices a and b
    def cut(self,a,b):
        if not(a<b):
            a=0
            b=self.len()
        self.dta=self.dta[:,a:b]
        self.x0=a*self.dx

    # append a data array
    def append(self,arr):
        self.dta=np.hstack((self.dta,arr))

    # scale data array
    def scale(self,f):
        self.dta=self.dta*f

    # export in spreadsheet format
    def export(self,filename,disp=None):
        if disp==None:
            dt=np.vstack([self.xvect(),self.dta])
        else:
            dt=np.vstack([self.xvect(),self.dta[disp]])
        np.savetxt(filename,dt.transpose(),delimiter=',')

    # save in binary format. file = open file handle or filename
    def save(self,file):
        self.dta.astype('float64').tofile(file)

    # convert data to log
    def to_log(self):
        self.dta=np.log10(self.dta)


# time domain data
class timedata(data):
    def __init__(self, dta=[], dx=1, x0=0, nskip=0):
        super().__init__(dta,dx,x0,nskip)

    # plot into a graph with predefined axes
    def plot(self,ax,xlabel='Time [s]',disp=None):
        if disp is None:
            disp=np.arange(self.nchann())
        ax.clear()
        for yind in disp:
            try:
                y=self.dta[yind]
                pl,=ax.plot(self.xvect(),y)
            except:
                pass
        ax.set_xlabel(xlabel)
        pl.figure.set_tight_layout('pad')
        pl.figure.canvas.draw()

    # export audio file
    def exportaudio(self,filename,trace=0):
        rate=np.int32(1/self.dx)
        scaled = np.int16(self.dta[trace] / np.max(np.abs(self.dta[trace])) * np.iinfo(np.int16).max)
        write(filename, rate, scaled)

    # exponential window function
    def expwd(self,r=1,a=0,b=0):
        #a,b=self.ibounds(a,b)
        le=b-a
        t=np.linspace(0,le*self.dx,le,endpoint=False)
        w=np.exp(-r*t)
        return(w)

    # symmetric exponential window
    def symexpwd(self,r=1,a=0,b=0):
        le=b-a
        t=np.linspace(0,le*self.dx,le,endpoint=False)
        t0=(self.ind_to_x(b)-self.ind_to_x(a))/2
        w=np.exp(-r*np.abs(t-t0))
        return(w)

    # gauss window
    def gausswd(self,r=3,a=0,b=0):
        le=b-a
        x=np.linspace(-r,r,le,endpoint=True)
        w=np.exp(-x**2)
        return(w)

    # sine square window
    def sinwd(self,r=1,a=0,b=0):
        le=b-a
        x=np.linspace(0,np.pi,le,endpoint=True)
        w=np.sin(x)**r
        return(w)

    # perform Fourier transform
    # x1,x2: bounds in units of the time domain data. Used if fc=0, overruled if fc!=0
    # fc: bounds as percentage of time domain data from center. Full domain if fc=1
    # ph: phase
    # wdw: window function ("exp", "symexp", "gauss", "sin", None)
    # r: window function parameter
    def ft(self,x1=0,x2=0,ph=None,wdw=None,r=1,fc=0):
        if(self.len()==0):
            ret=freqdata(np.array([]),1)
        else:
            if fc==0:
                a,b=self.ibounds(x1,x2)
            else:
                a,b=self.fcbounds(fc)
            le=b-a
            nm=2/le   # normalization factor
            if wdw=="exp":
                w=self.expwd(r,a,b)
                dt=self.dta[:,a:b]*w
            elif wdw=="symexp":
                w=self.symexpwd(r,a,b)
                dt=self.dta[:,a:b]*w
            elif wdw=="gauss":
                w=self.gausswd(r,a,b)
                dt=self.dta[:,a:b]*w
            elif wdw=="sin":
                w=self.sinwd(r,a,b)
                dt=self.dta[:,a:b]*w
            else:
                dt=self.dta[:,a:b]
            if ph==None:
                s=np.abs(np.fft.fft(dt))*nm  # using absolute value of ft
            else:
                dt=dt*np.exp(1j*ph)
                if(fc>0):
                    dt=np.fft.fftshift(dt)   #symmetric ft
                s=np.real(np.fft.fft(dt)*nm)
            m=round(le/2)
            s=s[:,0:m]
            ret=freqdata(s,1/self.dx/le)
        return(ret)

    # 4th order Butterworth bandpass filter of time domain data
    def butter(self,bandpass=[1000,50000]):
        b_b, b_a=butter(4,bandpass,btype='bandpass',fs=1/self.dx)
        self.dta=filtfilt(b_b,b_a,self.dta,axis=1)

# frequency domain data
class freqdata(data):
    def __init__(self, dta=[], dx=1):
        super().__init__(dta,dx)

    # plot into a graph with predefined axes
    def plot(self,ax,xlabel='Frequency [Hz]',disp=None):
        if disp is None:
            disp=np.arange(self.nchann())
        ax.clear()
        for yind in disp:
            try:
                y=self.dta[yind]
                pl,=ax.plot(self.xvect(),y)
            except:
                pass
        ax.set_xlabel(xlabel)
        pl.figure.set_tight_layout('pad')
        pl.figure.canvas.draw()

    # calculate signal-to-noise ratio, s1...s2 signal region frequencies, n1...n2 noise region frequencies
    # sind [index of data trace], ph [None=magnitude spectrum, otherwise phased spectrum]
    def calsnr(self,s1,s2,n1,n2,sind=0,ph=None):
        s1i,s2i=self.ibounds(s1,s2)
        n1i,n2i=self.ibounds(n1,n2)
        if ph==None:   #magnitude spectrum
            sig=np.max(self.dta[sind,s1i:s2i])
            nois=(np.sum(np.abs(self.dta[sind,n1i:n2i])**2)/(n2i-n1i))**0.5
        else:   #phased spectrum
            bsl=np.mean(self.dta[sind,n1i:n2i])
            sig=np.max(self.dta[sind,s1i:s2i])-bsl
            nois=np.std(self.dta[sind,n1i:n2i])
            #nois=(np.sum(np.abs(self.dta[sind,n1i:n2i]-bsl)**2)/(n2i-n1i))**0.5  # same as using std
        return(sig/nois)

    # calculate integrals
    def integrate(self, f1, f2):
        n1,n2=self.ibounds(f1,f2)
        ints=np.sum(self.dta[:,n1:n2],axis=1)
        return(ints)

class waveform(timedata):
    def __init__(self, dta=[], dx=1, x0=0, nskip=0):
        super().__init__(dta,dx,x0,nskip)

    # add pulse. phase is in multiples of pi
    # --handling of last point changed--
    def pulse(self,tstart=0,tlen=0.01,amp=1,frq=1000,phase=0,ch=0, endpoint=False):
        istart=self.x_to_ind(tstart)
        iend=self.x_to_ind(tstart+tlen)
        if endpoint:
            iend=iend+1
        #print((istart,iend))
        for i in range(istart,iend):
            self.dta[ch,i]=self.dta[ch,i]+amp*np.sin(2*np.pi*frq*self.ind_to_x(i)+np.pi/2*phase)
        return(tstart+tlen)

    # add constant voltage (dc) pulse
    def dcpulse(self,tstart=0,tlen=0.01,amp=1,ch=0, endpoint=False):
        istart=self.x_to_ind(tstart)
        iend=self.x_to_ind(tstart+tlen)
        if endpoint:
            iend=iend+1
        for i in range(istart,iend):
            self.dta[ch,i]=self.dta[ch,i]+amp
        return(tstart+tlen)

    # add dipsi2 sequence.
    # phase in multiples of pi, n=number of 4x supercycles, pw=90 deg pulse length
    # returns time length of sequence
    def dipsi2(self,tstart=0,pw=0.001,n=1,amp=1,frq=1000,phase=0,ch=0,nsuper=4):
        tlen=np.array([320, 410, 290, 285, 30, 245, 375, 265, 370])/90*pw
        phs=np.array([1,-1,1,-1,1,-1,1,-1,1])+phase
        supercycle=[0,2,2,0]
        if(nsuper<1):
            nsuper=1
        if(nsuper>4):
            nsuper=4
        sc2=supercycle[0:nsuper]
        t=tstart
        for k in range(n):
            for s in sc2:
                for _,(tl,p) in enumerate(zip(tlen,phs)):
                    self.pulse(tstart=t,tlen=tl,amp=amp,frq=frq,phase=p+s,ch=ch,endpoint=False)
                    t=t+tl
        return(t)

    #replace time array with zero array of specified length
    #t [time length /s], nchann [number of channels]
    def zero(self, t=1, nchann=1):
        n=self.x_to_ind(t,bounds=False)
        self.dta=np.zeros((nchann,n))

    #generate waveform tapered with sin squared function at beginning and end.
    def sin2taperpulse(self, tstart=0, t=1, rampt=0.01, amp=1, ch=0):
        istart=self.x_to_ind(tstart)
        nsamp=self.x_to_ind(t)
        nsramp=self.x_to_ind(rampt)
        r=np.sin(np.linspace(0,np.pi/2,nsramp,endpoint=False))**2
        on=np.ones(nsamp-2*nsramp)
        f=amp*np.concatenate((r,on,np.flip(r)))
        self.dta[ch,istart:istart+nsamp]=f

class integraldata(data):
    def __init__(self, dta=[], dx=1, x0=0, nskip=0):
        super().__init__(dta,dx,x0,nskip)

    # plot into a graph with predefined axes
    def plot(self,ax,xlabel='x',disp=None):
        if disp is None:
            disp=np.arange(self.nchann())
        ax.clear()
        for yind in disp:
            try:
                y=self.dta[yind]
                pl,=ax.plot(self.xvect(),y)
            except:
                pass
        ax.set_xlabel(xlabel)
        pl.figure.set_tight_layout('pad')
        pl.figure.canvas.draw()
