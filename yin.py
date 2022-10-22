'''compares autocorrelation, difference function, and 
cumulative mean normalized difference function for pitch detection
Source: https://www.youtube.com/watch?v=W585xR3bjLM'''


from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    f_0 = 1
    envelope = lambda x: np.exp(-x)
    return envelope(x) * np.sin(x * np.pi * 2 * f_0) 

def ACF(f, W, t, lag):
    return np.sum(f[t : t + W] * f[t + lag : t + lag + W])

def detect_pitch(f, W, t, sample_rate, bounds, thresh=0.1):
    CMNDF_vals = [CMNDF(f, W, t, i) for i in range(*bounds)]
    sample = None
    for i, val in enumerate(CMNDF_vals):
        if val < thresh:
            sample = i + bounds[0]
            break
        if sample is None:
            sample = np.argmin(CMNDF_vals) + bounds[0]
    return sample_rate / sample

def DF(f, W, t, lag):
    return ACF(f, W, t, 0)\
        + ACF(f, W, t + lag, 0)\
        - (2 * ACF(f, W, t, lag)) 
        
def CMNDF(f, W, t, lag):
    if lag == 0:
        return 1
    return DF(f, W, t, lag)\
        / np.sum([DF(f, W, t, j + 1) for j in range(lag)]) * lag 

if __name__ == "__main__":
    # sample_rate, data = wavfile.read("singer.wav")
    # data = data.astype(np.float64)
    # window_size = int(5 / 2000 * 44100)
    # bounds = [20, 2000]
    '''
    pithces = []
    for i in range(data.shape[0] // window_size + 3):
        pitches.append(
            detect_pitch(
                data,
                window_size,
                i * window_size, 
                sample_rate,
                bounds
                )
            )
    '''
    sample_rate = 50                                        # 500
    start = 0
    end = 5
    num_samples = int(sample_rate * (end - start) + 1)
    window_size = 20                                        # 200
    bounds = [20, num_samples // 2]   
    x = np.linspace(start, end, num_samples)
    t = 1
    W = 20
    lag = 1   
    
###################### PLOT 0 ######################
    fig, ax = plt.subplots(2,4, figsize=(24,10), dpi=300)
    ax[0,0].plot(x, f(x), '.', c='k') 
    ax[0,0].plot(x[t:t+W], f(x)[t:t+W], c='k')   
    acf = []
    dff = []
    cmndf = []
    lacf = []
    for l in range (1, 102, 10):
        l_ = l*0.1       
        ax[0,0].plot(x[t+l:t+l+W], f(x)[t+l:t+l+W])       
        ax[1,0].plot(l_+f(x)[t:t+W], c='k')                 # Window Segment Reference
        ax[1,0].plot(l_+f(x)[t+l:t+l+W])                    # Window Segment Lag  
        ax[0,1].plot(l_+f(x)[t:t+W]*f(x)[t+l:t+l+W])        # Product for ACF Calculation 
        ax[1,1].plot(l, ACF(f(x), W, t, l), '*')            # ACF Calculated
        ax[0,2].plot(l_+f(x)[t:t+W]-f(x)[t+l:t+l+W])        # Difference
        ax[1,2].plot(l,DF(f(x), W, t, l),'*')               # DF Calculated
        ax[0,3].plot(l_+f(x)[t:t+W]-f(x)[t+l:t+l+W])        # Difference
        ax[1,3].plot(l,CMNDF(f(x), W, t, l),'*')            # CMNDF Calculated
        lacf.append(l)
        acf.append(np.sum(f(x)[t:t+W]*f(x)[t+l:t+l+W])) 
        dff.append(DF(f(x), W, t, l)) 
        cmndf.append(CMNDF(f(x), W, t, l)) 
    
    ax[1,1].plot(lacf,acf, 'k')
    ax[1,2].plot(lacf,dff, 'k')
    ax[1,3].plot(lacf,cmndf, 'k')
    
    ax[0,0].set_title('Signal')
    ax[0,0].set_xlabel('t (sec)')
    ax[0,1].set_title('Product')
    ax[0,1].set_xlabel('n (sample)')
    ax[0,2].set_title('Difference')
    ax[0,2].set_xlabel('n (sample)')
    ax[0,3].set_title('Difference')
    ax[0,3].set_xlabel('n (sample)')
    ax[1,0].set_title('Window')
    ax[1,0].set_xlabel('n (sample)')
    ax[1,1].set_title('ACF')
    ax[1,1].set_xlabel('lag (sample)')
    ax[1,2].set_title('DF')
    ax[1,2].set_xlabel('lag (sample)')
    ax[1,3].set_title('CMNDF')
    ax[1,3].set_xlabel('lag (sample)')
 
    print(detect_pitch(f(x), window_size, 1, sample_rate, bounds))  
    plt.savefig('pitch.jpeg', dpi=300)
    
    
