
class Parameters(dict):
    
    def __init__(self):
        self['max_drift'] = 10
        self['savefile_prefix']=''
        self['fft_size']=512
        self['window_size']=128
        self['step_size']=32
        self['q_bandwidth']=0.01
        self['th_bandwidth']=0.01
        self['threads']=8
        self['processor']='cpu'
        self['q']=0
        self['q_low']=0
        self['q_high']=0
        
        

    