import numpy as np

FunctionLibrary = {}

try:
    import scipy as sp

    def BandPassFilter(signal=0, order=3, low_cutoff=10, high_cutoff=20, t=0):
        sampling = np.quantile(np.diff(t), 0.5)
        cutoff = sorted([low_cutoff[0], high_cutoff[0]])
        fs = 1 /  sampling
        order = order[0]
        sos = sp.signal.butter(order, cutoff, btype="bandpass", fs=fs, output="sos")
        return sp.signal.sosfilt(sos, signal)
    
    def BandStopFilter(signal=0, order=3, low_cutoff=10, high_cutoff=20, t=0):
        sampling = np.quantile(np.diff(t), 0.5)
        cutoff = sorted([low_cutoff[0], high_cutoff[0]])
        fs = 1 /  sampling
        order = order[0]
        sos = sp.signal.butter(order, cutoff, btype="bandstop", fs=fs, output="sos")
        return sp.signal.sosfilt(sos, signal)
    
    def LowPassFilter(signal=0, order=3, cutoff=10, t=0):
        sampling = np.quantile(np.diff(t), 0.5)
        fs = 1 /  sampling
        order = order[0]
        cutoff = cutoff[0]
        sos = sp.signal.butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
        return sp.signal.sosfilt(sos, signal)
    
    def HighPassFilter(signal=0, order=3, cutoff=10, t=0):
        sampling = np.quantile(np.diff(t), 0.5)
        fs = 1 /  sampling
        order = order[0]
        cutoff = cutoff[0]
        sos = sp.signal.butter(order, cutoff, btype="highpass", fs=fs, output="sos")
        return sp.signal.sosfilt(sos, signal)
    
    FunctionLibrary["BandPassFilter"] = BandPassFilter
    FunctionLibrary["BandStopFilter"] = BandStopFilter
    FunctionLibrary["LowPassFilter"] = LowPassFilter
    FunctionLibrary["HighPassFilter"] = HighPassFilter

except ImportError:
    pass