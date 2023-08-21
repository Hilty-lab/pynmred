# (c) Hilty Lab, 2022-2023

import numpy as np
from .expdta import waveform

class pulsebase(waveform):
    def __init__(self, p):   # duplicate of the acquisition parameters from the experiment in p
        dta=np.zeros(p["nsamp"])
        super().__init__(dta,1/p['srate'])
        self.p=p
        self.genpp()

    def genpp(self):
        if self.p['ppg']=='cpmg':
            self.cpmg_pp()
        elif self.p['ppg']=='pwecho':
            self.pwecho_pp()
        elif self.p['ppg']=='cpmgdual':
            self.cpmgdual_pp()
        elif self.p['ppg']=='cpmgtrigg':
            self.cpmgtrigg_pp()
        elif self.p['ppg']=='cpmgt1trigg':
            self.cpmgt1trigg_pp()
        elif self.p['ppg']=='cpmgdualtrigg':
            self.cpmgdualtrigg_pp()
        elif self.p['ppg']=='r1':
            self.r1_pp()
        elif self.p['ppg']=='cnst':
            self.cnst_pp()
        elif self.p['ppg']=='tune':
            self.tune_pp()
        elif self.p['ppg']=='travel':
            self.travel_pp()
        elif self.p['ppg']=='traveltrigg':
            self.traveltrigg_pp()
        elif self.p['ppg']=='dipsi2':
            self.dipsi2_pp()
        elif self.p['ppg']=='dipsi2dual':
            self.dipsi2dual_pp()
        elif self.p['ppg']=='dipsi2dualstp':
            self.dipsi2dualstp_pp()
        elif self.p['ppg']=='diff':
            self.diff_pp()
        else:
            raise Exception("Pulse program not found")

    # Pulse programs
    #
    # Common parameters:
    # nsamp [number of samples], srate [sample rate /Hz], timeout [acquisition timeout /s]
    # inchann [input channel of form /Dev1/ai0]
    # inclk [sample clock of form /Dev1/ao/SampleClock; none for boards without internal synchronization]
    # outchann [output channel of form /Dev1/ao0]
    # pp [pulse program name]
    # aitrigg [input line to trigger analog input, of form /Dev1/PFI0]
    # aotrigg [input line to trigger analog output, of form /Dev1/PFI1]
    # triggdo [output line to send a trigger signal, of fomr /Dev1/port0/line0]
    # nskip [acquired data points to skip]

    # cpmg pulse program
    # Additional parameters:
    # p1 [pulse length /s], amp [pulse amplitude /V], frq [pulse frequency /Hz], necho [number of echoes], tau [half echo time /s]
    def cpmg_pp(self):
        self.pulse(tstart=0,tlen=self.p['p1'],amp=self.p['amp'],frq=self.p['frq'],phase=self.p['p1phase'])
        for i in range(self.p['necho']):
            self.pulse(tstart=self.p['p1']+self.p['tau']+2*i*(self.p['p1']+self.p['tau']),tlen=2*self.p['p1'],amp=self.p['amp'],frq=self.p['frq'],phase=1)
        self.dta[0,-1]=0

    # pulselength determination with echo pulse program
    # Additional parameters:
    # p1 [pulse length /s], amp [pulse amplitude /V], frq [pulse frequency /Hz], tau [half echo time /s]
    # tcomp [compensation delay to place fid at position independent of t1]

    def pwecho_pp(self):
        tst=self.p['tcomp']-3*self.p['p1']  #fid position independent of t1
        self.pulse(tstart=tst,tlen=self.p['p1'],amp=self.p['amp'],frq=self.p['frq'],phase=self.p['p1phase'])
        self.pulse(tstart=tst+self.p['p1']+self.p['tau'],tlen=2*self.p['p1'],amp=self.p['amp'],frq=self.p['frq'],phase=1)
        self.dta[0,-1]=0

    # calculate parameters for split
    def cpmg_sp(self):
        ioff=self.x_to_ind(2*self.p['p1']+self.p['tau'])
        ilen=self.x_to_ind(2*self.p['p1']+2*self.p['tau'])
        return(ioff,ilen)

    # cpmg pulse program with dual-band pulses
    # Additional parameters:
    # p1 [pulse length /s], amp1/amp2 [pulse amplitude1/2 /V], frq1/2 [pulse frequency1/2 /Hz], necho [number of echoes], tau [half echo time /s]
    def cpmgdual_pp(self):
        self.pulse(tstart=0,tlen=self.p['p1'],amp=self.p['amp1'],frq=self.p['frq1'],phase=self.p['p1phase'])
        self.pulse(tstart=0,tlen=self.p['p1'],amp=self.p['amp2'],frq=self.p['frq2'],phase=self.p['p1phase'])
        for i in range(self.p['necho']):
            self.pulse(tstart=self.p['p1']+self.p['tau']+2*i*(self.p['p1']+self.p['tau']),tlen=2*self.p['p1'],amp=self.p['amp1'],frq=self.p['frq1'],phase=1)
            self.pulse(tstart=self.p['p1']+self.p['tau']+2*i*(self.p['p1']+self.p['tau']),tlen=2*self.p['p1'],amp=self.p['amp2'],frq=self.p['frq2'],phase=1)
        self.dta[0,-1]=0

    # cpmg pulse program with dual-band pulses and trigger
    # Additional parameters:
    # p1 [pulse length /s], amp1/amp2 [pulse amplitude1/2 /V], frq1/2 [pulse frequency1/2 /Hz], necho [number of echoes], tau [half echo time /s]
    # tt [trigger switch time], tp [pulse start time]    
    def cpmgdualtrigg_pp(self):
        self.p['amp2']=self.p['amp1']*1.06
        self.dta=np.zeros((2,self.p['nsamp']))
        self.dcpulse(tstart=self.p['tt'],tlen=self.xmax(),amp=5,ch=1,endpoint=True)
        po=self.p['tp']
        self.pulse(tstart=po,tlen=self.p['p1'],amp=self.p['amp1'],frq=self.p['frq1'],phase=self.p['p1phase'])
        self.pulse(tstart=po,tlen=self.p['p1'],amp=self.p['amp2'],frq=self.p['frq2'],phase=self.p['p1phase'])
        for i in range(self.p['necho']):
            self.pulse(tstart=po+self.p['p1']+self.p['tau']+2*i*(self.p['p1']+self.p['tau']),tlen=2*self.p['p1'],amp=self.p['amp1'],frq=self.p['frq1'],phase=1)
            self.pulse(tstart=po+self.p['p1']+self.p['tau']+2*i*(self.p['p1']+self.p['tau']),tlen=2*self.p['p1'],amp=self.p['amp2'],frq=self.p['frq2'],phase=1)
        self.dta[0,-1]=0
    
    # R1 relaxation pulse program
    # Additional parameters:
    # p1 [pulse length /s], amp [pulse amplitude /V], frq [pulse frequency /Hz], n [number of pulses], tau [measurement time /s]
    def r1_pp(self):
        for i in range(self.p['n']):
            self.pulse(tstart=self.p['p1']*i+self.p['tau']*i,tlen=self.p['p1'],amp=self.p['amp'],frq=self.p['frq'],phase=1)
        self.dta[0,-1]=0
        
    # pulse program for diffusion experiment
    # Additional parameters:
    # p1 [pulse length /s], amp1/amp2 [pulse amplitude1/2 /V], frq1/2 [pulse frequency1/2 /Hz], necho [number of echoes], tau [half echo time /s]
    def diff_pp(self):
        self.dta=np.zeros((2,self.p['nsamp']))
        tst=self.p['tcomp']-3*self.p['p1']  #fid position independent of t1
        self.pulse(tstart=tst,tlen=self.p['p1'],amp=self.p['amp1'],frq=self.p['frq'],phase=self.p['p1phase'],ch=0)
        self.dcpulse(tstart=tst+self.p['p1']+self.p['grd_delay'],tlen=self.p['p2'],amp=self.p['amp2'],ch=1)
        self.pulse(tstart=tst+self.p['p1']+self.p['tau'],tlen=2*self.p['p1'],amp=self.p['amp1'],frq=self.p['frq'],phase=1,ch=0)
        self.dcpulse(tstart=tst+3*self.p['p1']+self.p['tau']+self.p['grd_delay'],tlen=self.p['p2'],amp=self.p['amp2'],ch=1)
        self.dta[0,-1]=0
        self.dta[1,-1]=0

    # dipsi2 pulse program
    # Additional parameters:
    # phase [phase in multiples of pi], amp [pulse amplitude /V], frq [pulse frequency /Hz], n [number of dipsi2 supercycles], t1 [evolution before dipsi2]
    # pwa [90 degree pulse length before dipsi], pwd [90 degree pulse length for dipsi], pwb [90 degre pulse after dipsi], delta [pre/post dipsi delay]
    # tcomp [compensation delay to place fid at position independent of t1]
    def dipsi2_pp(self):
        tst=self.p['tcomp']-self.p['t1']  #fid position independent of t1
        self.pulse(tstart=tst,tlen=self.p['pwa'],amp=self.p['amp'],frq=self.p['frq'],phase=self.p['phase1'])
        self.pulse(tstart=tst+self.p['pwa']+self.p['t1'],tlen=self.p['pwa'],amp=self.p['amp'],frq=self.p['frq'],phase=0)
        td=self.dipsi2(tstart=tst+2*self.p['pwa']+self.p['t1']+self.p['delta'],pw=self.p['pwd'],n=self.p['n'],amp=self.p['amp'],frq=self.p['frq'],phase=0)
        self.pulse(tstart=td+self.p['delta'],tlen=self.p['pwb'],amp=self.p['amp'],frq=self.p['frq'],phase=0)
        self.dta[0,-1]=0

    # dipsi2 pulse program with dual excitation
    # Additional parameters:
    # phase [phase in multiples of pi], amp1/2 [pulse amplitude1/2 /V], frq1/2 [pulse frequency1/2 /Hz], n [number of dipsi2 supercycles], t1 [evolution before dipsi2]
    # pwa [90 degree pulse length before dipsi], pwd [90 degree pulse length for dipsi], pwb [90 degre pulse after dipsi], delta [pre/post dipsi delay]
    # tcomp [compensation delay to place fid at position independent of t1]
    def dipsi2dual_pp(self):
        tst=self.p['tcomp']-self.p['t1']  #fid position independent of t1
        self.pulse(tstart=tst,tlen=self.p['pwa'],amp=self.p['amp'],frq=self.p['frq'],phase=self.p['phase1'])

        self.pulse(tstart=tst+self.p['pwa']+self.p['t1'],tlen=self.p['pwa'],amp=self.p['amp'],frq=self.p['frq'],phase=0)

        td=self.dipsi2(tstart=tst+2*self.p['pwa']+self.p['t1']+self.p['delta'],pw=self.p['pwd'],n=self.p['n'],amp=self.p['amp1'],frq=self.p['frq1'],phase=0)
        td=self.dipsi2(tstart=tst+2*self.p['pwa']+self.p['t1']+self.p['delta'],pw=self.p['pwd'],n=self.p['n'],amp=self.p['amp2'],frq=self.p['frq2'],phase=0)

        self.pulse(tstart=td+self.p['delta'],tlen=self.p['pwb'],amp=self.p['amp'],frq=self.p['frq'],phase=0)

        self.dta[0,-1]=0
        
        
    # dipsi2 pulse program with dual excitation, states/tppi
    # Additional parameters:
    # phase [phase in multiples of pi], amp1/2 [pulse amplitude1/2 /V], frq1/2 [pulse frequency1/2 /Hz], n [number of dipsi2 supercycles], t1 [evolution before dipsi2]
    # pwa [90 degree pulse length before dipsi], pwd [90 degree pulse length for dipsi], pwb [90 degre pulse after dipsi], delta [pre/post dipsi delay]
    # tcomp [compensation delay to place fid at position independent of t1]
    # k [incremented scan number], dt1 [indirect dimension dwell time /s]
    def dipsi2dualstp_pp(self):
        nsuper=self.p.get('nsuper',4)
        dt1=self.p['dt1']
        t10=self.p['t1']
        k=self.p['k']
        t1=t10+k//2*dt1
        phase1=(self.p['phase1']+k)%4
        tst=self.p['tcomp']-t1  #fid position independent of t1
        self.pulse(tstart=tst,tlen=self.p['pwa'],amp=self.p['amp'],frq=self.p['frq'],phase=phase1)

        self.pulse(tstart=tst+self.p['pwa']+t1,tlen=self.p['pwa'],amp=self.p['amp'],frq=self.p['frq'],phase=0)

        td=self.dipsi2(tstart=tst+2*self.p['pwa']+t1+self.p['delta'],pw=self.p['pwd'],n=self.p['n'],amp=self.p['amp1'],frq=self.p['frq1'],phase=0,nsuper=nsuper)
        td=self.dipsi2(tstart=tst+2*self.p['pwa']+t1+self.p['delta'],pw=self.p['pwd'],n=self.p['n'],amp=self.p['amp2'],frq=self.p['frq2'],phase=0,nsuper=nsuper)

        self.pulse(tstart=td+self.p['delta'],tlen=self.p['pwb'],amp=self.p['amp'],frq=self.p['frq'],phase=0)

        self.dta[0,-1]=0
        

    # cpmg pulse program with trigger
    # Additional parameters:
    # p1 [pulse length /s], amp [pulse amplitude /V], frq [pulse frequency /Hz], necho [number of echoes], tau [half echo time /s]
    # tt [trigger switch time], tp [pulse start time]
    def cpmgtrigg_pp(self):
        self.dta=np.zeros((2,self.p['nsamp']))
        self.dcpulse(tstart=self.p['tt'],tlen=self.xmax(),amp=5,ch=1,endpoint=True)
        po=self.p['tp']
        self.pulse(tstart=po,tlen=self.p['p1'],amp=self.p['amp'],frq=self.p['frq'],phase=self.p['p1phase'])
        for i in range(self.p['necho']):
            self.pulse(tstart=po+self.p['p1']+self.p['tau']+2*i*(self.p['p1']+self.p['tau']),tlen=2*self.p['p1'],amp=self.p['amp'],frq=self.p['frq'],phase=1)
        self.dta[0,-1]=0

    # cpmg pulse program with t1 delay and trigger
    # Additional parameters:
    # p1 [pulse length /s], amp [pulse amplitude /V], frq [pulse frequency /Hz], necho [number of echoes], tau [half echo time /s]
    # t1 [t1 relaxation time between initial pi pulse and pi/2 pulse], pinvamp [amplitude for inversion pulse], pinvt [time for inversion pulse]
    # tt [trigger switch time], tp [pulse start time]
    def cpmgt1trigg_pp(self):
        self.dta=np.zeros((2,self.p['nsamp']))
        self.dcpulse(tstart=self.p['tt'],tlen=self.xmax(),amp=5,ch=1,endpoint=True)
        po=self.p['tp']
        self.pulse(tstart=po,tlen=self.p['pinvt'],amp=self.p['pinvamp'],frq=self.p['frq'],phase=0)
        po2=po+self.p['pinvt']+self.p['t1']
        self.pulse(tstart=po2,tlen=self.p['p1'],amp=self.p['amp'],frq=self.p['frq'],phase=self.p['p1phase'])
        for i in range(self.p['necho']):
            self.pulse(tstart=po2+self.p['p1']+self.p['tau']+2*i*(self.p['p1']+self.p['tau']),tlen=2*self.p['p1'],amp=self.p['amp'],frq=self.p['frq'],phase=1)
        self.dta[0,-1]=0

    # travel time / T1 pulse program
    # Additional parameters:
    # p180 [inversion pulse length /s], p1 [excitation pulse length /s], amp [pulse amplitude /V], frq [pulse frequency /Hz],
    #     npulse [number of excitation pulses], tau [time between pulses /s]
    def travel_pp(self):
        self.pulse(tstart=0,tlen=self.p['p180'],amp=self.p['amp'],frq=self.p['frq'],phase=1)
        for i in range(self.p['npulse']):
            self.pulse(tstart=self.p['p180']+i*self.p['tau'],tlen=self.p['p1'],amp=self.p['amp'],frq=self.p['frq'],phase=1)
        self.dta[0,-1]=0

    # travel time / T1 pulse program with trigger
    # Additional parameters:
    # p180 [inversion pulse length /s], p1 [excitation pulse length /s], amp [pulse amplitude /V], frq [pulse frequency /Hz],
    #     npulse [number of excitation pulses], tau [time between pulses /s]
    # tt [trigger switch time], tp [pulse start time]
    def traveltrigg_pp(self):
        self.dta=np.zeros((2,self.p['nsamp']))
        self.dcpulse(tstart=self.p['tt'],tlen=self.xmax(),amp=5,ch=1,endpoint=True)
        po=self.p['tp']
        self.pulse(tstart=po,tlen=self.p['p180'],amp=self.p['amp'],frq=self.p['frq'],phase=1)
        for i in range(self.p['npulse']):
            self.pulse(tstart=po+self.p['p180']+i*self.p['tau'],tlen=self.p['p1'],amp=self.p['amp'],frq=self.p['frq'],phase=1)
        self.dta[0,-1]=0

    # constant pulse program
    # Additional parameters:
    # amp [pulse amplitude /V]
    def cnst_pp(self):
        self.dta=np.array([np.ones(self.len())*self.p['amp']])
        self.dta[0,-1]=0

    # pulse program for tuning
    def tune_pp(self):
        self.dcpulse(tstart=0,tlen=self.p['p1'],amp=self.p['amp'])
        self.dta[0,-1]=0
