import numpy as np

FunctionLibrary = {}

try:
    import scipy as sp

    def BandPassFilter_Butter(signal=0, order=3, low_cutoff=10, high_cutoff=20, t=0):
        """
        Apply a band pass Butterworth filter on the input signal. 
        This function can only be used in complete signal mode.

        Parameters
        ----------
        signal : input signal

        order : int
            the order of the filter; default 3

        low_cutoff : float
            low cut off frequency in Hz; default 10

        high_cutoff : float
            high cut off frequency in Hz; default 20

        t : signal timestamps


        Returns
        -------
        filtered signal
        """
        sampling = np.quantile(np.diff(t), 0.5)
        cutoff = sorted([low_cutoff[0], high_cutoff[0]])
        fs = 1 /  sampling
        order = order[0]
        sos = sp.signal.butter(order, cutoff, btype="bandpass", fs=fs, output="sos")
        return sp.signal.sosfilt(sos, signal)
    
    def BandPassFilter_Cheby1(signal=0, order=3, max_ripple=5, low_cutoff=10, high_cutoff=20, t=0):
        """
        Apply a band pass Chebyshev type I filter on the input signal. 
        This function can only be used in complete signal mode.

        Parameters
        ----------
        signal : input signal

        order : int
            the order of the filter; default 3

        max_ripple: float
            The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number. default 5

        low_cutoff : float
            low cut off frequency in Hz; default 10

        high_cutoff : float
            high cut off frequency in Hz; default 20

        t : signal timestamps


        Returns
        -------
        filtered signal
        """
        sampling = np.quantile(np.diff(t), 0.5)
        cutoff = sorted([low_cutoff[0], high_cutoff[0]])
        fs = 1 /  sampling
        order = order[0]
        max_ripple = max_ripple[0]
        sos = sp.signal.cheby1(order, max_ripple, cutoff, btype="bandpass", fs=fs, output="sos")
        return sp.signal.sosfilt(sos, signal)
    
    def BandPassFilter_Cheby2(signal=0, order=3, min_attenuation=40, low_cutoff=10, high_cutoff=20, t=0):
        """
        Apply a band pass Chebyshev type II filter on the input signal. 
        This function can only be used in complete signal mode.

        Parameters
        ----------
        signal : input signal

        order : int
            the order of the filter; default 3

        min_attenuation: float
            The minimum attenuation required in the stop band. Specified in decibels, as a positive number. default 40

        low_cutoff : float
            low cut off frequency in Hz; default 10

        high_cutoff : float
            high cut off frequency in Hz; default 20

        t : signal timestamps


        Returns
        -------
        filtered signal
        """
        sampling = np.quantile(np.diff(t), 0.5)
        cutoff = sorted([low_cutoff[0], high_cutoff[0]])
        fs = 1 /  sampling
        order = order[0]
        min_attenuation = min_attenuation[0]
        sos = sp.signal.cheby2(order, min_attenuation, cutoff, btype="bandpass", fs=fs, output="sos")
        return sp.signal.sosfilt(sos, signal)
    
    def BandPassFilter_Ellip(signal=0, order=3, max_ripple=5, min_attenuation=40, low_cutoff=10, high_cutoff=20, t=0):
        """
        Apply a band pass Eliptical filter on the input signal. 
        This function can only be used in complete signal mode.

        Parameters
        ----------
        signal : input signal

        order : int
            the order of the filter; default 3

        max_ripple: float
            The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number. default 5

        min_attenuation: float
            The minimum attenuation required in the stop band. Specified in decibels, as a positive number. default 40

        low_cutoff : float
            low cut off frequency in Hz; default 10

        high_cutoff : float
            high cut off frequency in Hz; default 20

        t : signal timestamps


        Returns
        -------
        filtered signal
        """
        sampling = np.quantile(np.diff(t), 0.5)
        cutoff = sorted([low_cutoff[0], high_cutoff[0]])
        fs = 1 /  sampling
        order = order[0]
        min_attenuation = min_attenuation[0]
        max_ripple = max_ripple[0]
        sos = sp.signal.ellip(order, max_ripple, min_attenuation, cutoff, btype="bandpass", fs=fs, output="sos")
        return sp.signal.sosfilt(sos, signal)
    
    def BandStopFilter_Butter(signal=0, order=3, low_cutoff=10, high_cutoff=20, t=0):
        """
        Apply a band stop Butterworth filter on the input signal. 
        This function can only be used in complete signal mode.

        Parameters
        ----------
        signal : input signal

        order : int
            the order of the filter; default 3

        low_cutoff : float
            low cut off frequency in Hz; default 10

        high_cutoff : float
            high cut off frequency in Hz; default 20

        t : signal timestamps


        Returns
        -------
        filtered signal
        """
        sampling = np.quantile(np.diff(t), 0.5)
        cutoff = sorted([low_cutoff[0], high_cutoff[0]])
        fs = 1 /  sampling
        order = order[0]
        sos = sp.signal.butter(order, cutoff, btype="bandstop", fs=fs, output="sos")
        return sp.signal.sosfilt(sos, signal)
    
    def BandStopFilter_Cheby1(signal=0, order=3, max_ripple=5, low_cutoff=10, high_cutoff=20, t=0):
        """
        Apply a band stop Chebyshev type I filter on the input signal. 
        This function can only be used in complete signal mode.

        Parameters
        ----------
        signal : input signal

        order : int
            the order of the filter; default 3

        max_ripple: float
            The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number. default 5

        low_cutoff : float
            low cut off frequency in Hz; default 10

        high_cutoff : float
            high cut off frequency in Hz; default 20

        t : signal timestamps


        Returns
        -------
        filtered signal
        """
        sampling = np.quantile(np.diff(t), 0.5)
        cutoff = sorted([low_cutoff[0], high_cutoff[0]])
        fs = 1 /  sampling
        order = order[0]
        max_ripple = max_ripple[0]
        sos = sp.signal.cheby1(order, max_ripple, cutoff, btype="bandstop", fs=fs, output="sos")
        return sp.signal.sosfilt(sos, signal)
    
    def BandStopFilter_Cheby2(signal=0, order=3, min_attenuation=40, low_cutoff=10, high_cutoff=20, t=0):
        """
        Apply a band stop Chebyshev type II filter on the input signal. 
        This function can only be used in complete signal mode.

        Parameters
        ----------
        signal : input signal

        order : int
            the order of the filter; default 3

        min_attenuation: float
            The minimum attenuation required in the stop band. Specified in decibels, as a positive number. default 40

        low_cutoff : float
            low cut off frequency in Hz; default 10

        high_cutoff : float
            high cut off frequency in Hz; default 20

        t : signal timestamps


        Returns
        -------
        filtered signal
        """
        sampling = np.quantile(np.diff(t), 0.5)
        cutoff = sorted([low_cutoff[0], high_cutoff[0]])
        fs = 1 /  sampling
        order = order[0]
        min_attenuation = min_attenuation[0]
        sos = sp.signal.cheby2(order, min_attenuation, cutoff, btype="bandstop", fs=fs, output="sos")
        return sp.signal.sosfilt(sos, signal)
    
    def BandStopFilter_Ellip(signal=0, order=3, max_ripple=5, min_attenuation=40, low_cutoff=10, high_cutoff=20, t=0):
        """
        Apply a band stop Eliptical filter on the input signal. 
        This function can only be used in complete signal mode.

        Parameters
        ----------
        signal : input signal

        order : int
            the order of the filter; default 3

        max_ripple: float
            The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number. default 5

        min_attenuation: float
            The minimum attenuation required in the stop band. Specified in decibels, as a positive number. default 40

        low_cutoff : float
            low cut off frequency in Hz; default 10

        high_cutoff : float
            high cut off frequency in Hz; default 20

        t : signal timestamps


        Returns
        -------
        filtered signal
        """
        sampling = np.quantile(np.diff(t), 0.5)
        cutoff = sorted([low_cutoff[0], high_cutoff[0]])
        fs = 1 /  sampling
        order = order[0]
        min_attenuation = min_attenuation[0]
        max_ripple = max_ripple[0]
        sos = sp.signal.ellip(order, max_ripple, min_attenuation, cutoff, btype="bandstop", fs=fs, output="sos")
        return sp.signal.sosfilt(sos, signal)

    def ClipSignal(signal=0, lower=0, upper=100, t=0):
        """
        Clip the signal to the lower and upper limits.
        This function can be used both complete and sample by sample modes.

        Parameters
        ----------
        signal : input signal

        lower : float
            lower limit; default 0. If the value is set to the string None, then the lower limit clipping is not performed. 

        upper : float
            upper limit; default 100. If the value is set to the string None, then the upper limit clipping is not performed.

        t : signal timestamps


        Returns
        -------
        clipped signal
        """
        try:
            lower = next(iter(lower))
        except TypeError:
            pass

        try:
            upper = next(iter(upper))
        except TypeError:
            pass

        return np.clip(signal, lower, upper)
    
    def LowPassFilter_Butter(signal=0, order=3, cutoff=10, t=0):
        """
        Apply a low pass Butterworth filter on the input signal. 
        This function can only be used in complete signal mode.

        Parameters
        ----------
        signal : input signal

        order : int
            the order of the filter; default 3

        cutoff : float
            cut off frequency in Hz; default 10

        t : signal timestamps


        Returns
        -------
        filtered signal
        """
        sampling = np.quantile(np.diff(t), 0.5)
        fs = 1 /  sampling
        order = order[0]
        cutoff = cutoff[0]
        sos = sp.signal.butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
        return sp.signal.sosfilt(sos, signal)
    
    def LowPassFilter_Cheby1(signal=0, order=3, max_ripple=5, cutoff=10, t=0):
        """
        Apply a low pass Chebyshev type I filter on the input signal. 
        This function can only be used in complete signal mode.

        Parameters
        ----------
        signal : input signal

        order : int
            the order of the filter; default 3

        max_ripple: float
            The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number. default 5

        cutoff : float
            cut off frequency in Hz; default 10

        t : signal timestamps

        
        Returns
        -------
        filtered signal
        """
        sampling = np.quantile(np.diff(t), 0.5)
        cutoff = cutoff[0]
        fs = 1 /  sampling
        order = order[0]
        max_ripple = max_ripple[0]
        sos = sp.signal.cheby1(order, max_ripple, cutoff, btype="lowpass", fs=fs, output="sos")
        return sp.signal.sosfilt(sos, signal)
    
    def LowPassFilter_Cheby2(signal=0, order=3, min_attenuation=40, cutoff=10, t=0):
        """
        Apply a low pass Chebyshev type II filter on the input signal. 
        This function can only be used in complete signal mode.

        Parameters
        ----------
        signal : input signal

        order : int
            the order of the filter; default 3

        min_attenuation: float
            The minimum attenuation required in the stop band. Specified in decibels, as a positive number. default 40

        cutoff : float
            cut off frequency in Hz; default 10

        t : signal timestamps


        Returns
        -------
        filtered signal
        """
        sampling = np.quantile(np.diff(t), 0.5)
        cutoff = cutoff[0]
        fs = 1 /  sampling
        order = order[0]
        min_attenuation = min_attenuation[0]
        sos = sp.signal.cheby2(order, min_attenuation, cutoff, btype="lowpass", fs=fs, output="sos")
        return sp.signal.sosfilt(sos, signal)
    
    def LowPassFilter_Ellip(signal=0, order=3, max_ripple=5, min_attenuation=40, low_cutoff=10, high_cutoff=20, t=0):
        """
        Apply a low pass Eliptical filter on the input signal. 
        This function can only be used in complete signal mode.

        Parameters
        ----------
        signal : input signal

        order : int
            the order of the filter; default 3

        max_ripple: float
            The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number. default 5

        min_attenuation: float
            The minimum attenuation required in the stop band. Specified in decibels, as a positive number. default 40

        low_cutoff : float
            low cut off frequency in Hz; default 10

        high_cutoff : float
            high cut off frequency in Hz; default 20

        t : signal timestamps


        Returns
        -------
        filtered signal
        """
        sampling = np.quantile(np.diff(t), 0.5)
        cutoff = sorted([low_cutoff[0], high_cutoff[0]])
        fs = 1 /  sampling
        order = order[0]
        min_attenuation = min_attenuation[0]
        max_ripple = max_ripple[0]
        sos = sp.signal.ellip(order, max_ripple, min_attenuation, cutoff, btype="lowpass", fs=fs, output="sos")
        return sp.signal.sosfilt(sos, signal)
    
    def HighPassFilter_Butter(signal=0, order=3, cutoff=10, t=0):
        """
        Apply a high pass Butterworth filter on the input signal. 
        This function can only be used in complete signal mode.

        Parameters
        ----------
        signal : input signal

        order : int
            the order of the filter; default 3

        cutoff : float
            cut off frequency in Hz; default 10

        t : signal timestamps


        Returns
        -------
        filtered signal
        """
        sampling = np.quantile(np.diff(t), 0.5)
        fs = 1 /  sampling
        order = order[0]
        cutoff = cutoff[0]
        sos = sp.signal.butter(order, cutoff, btype="highpass", fs=fs, output="sos")
        return sp.signal.sosfilt(sos, signal)
    
    def HighPassFilter_Cheby1(signal=0, order=3, max_ripple=5, cutoff=10, t=0):
        """
        Apply a high pass Chebyshev type I filter on the input signal. 
        This function can only be used in complete signal mode.

        Parameters
        ----------
        signal : input signal

        order : int
            the order of the filter; default 3

        max_ripple: float
            The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number. default 5

        cutoff : float
            cut off frequency in Hz; default 10

        t : signal timestamps

        
        Returns
        -------
        filtered signal
        """
        sampling = np.quantile(np.diff(t), 0.5)
        cutoff = cutoff[0]
        fs = 1 /  sampling
        order = order[0]
        max_ripple = max_ripple[0]
        sos = sp.signal.cheby1(order, max_ripple, cutoff, btype="highpass", fs=fs, output="sos")
        return sp.signal.sosfilt(sos, signal)
    
    def HighPassFilter_Cheby2(signal=0, order=3, min_attenuation=40, cutoff=10, t=0):
        """
        Apply a high pass Chebyshev type II filter on the input signal. 
        This function can only be used in complete signal mode.

        Parameters
        ----------
        signal : input signal

        order : int
            the order of the filter; default 3

        min_attenuation: float
            The minimum attenuation required in the stop band. Specified in decibels, as a positive number. default 40

        cutoff : float
            cut off frequency in Hz; default 10

        t : signal timestamps


        Returns
        -------
        filtered signal
        """
        sampling = np.quantile(np.diff(t), 0.5)
        cutoff = cutoff[0]
        fs = 1 /  sampling
        order = order[0]
        min_attenuation = min_attenuation[0]
        sos = sp.signal.cheby2(order, min_attenuation, cutoff, btype="highpass", fs=fs, output="sos")

    def HighPassFilter_Ellip(signal=0, order=3, max_ripple=5, min_attenuation=40, low_cutoff=10, high_cutoff=20, t=0):
        """
        Apply a high pass Eliptical filter on the input signal. 
        This function can only be used in complete signal mode.

        Parameters
        ----------
        signal : input signal

        order : int
            the order of the filter; default 3

        max_ripple: float
            The maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number. default 5

        min_attenuation: float
            The minimum attenuation required in the stop band. Specified in decibels, as a positive number. default 40

        low_cutoff : float
            low cut off frequency in Hz; default 10

        high_cutoff : float
            high cut off frequency in Hz; default 20

        t : signal timestamps


        Returns
        -------
        filtered signal
        """
        sampling = np.quantile(np.diff(t), 0.5)
        cutoff = sorted([low_cutoff[0], high_cutoff[0]])
        fs = 1 /  sampling
        order = order[0]
        min_attenuation = min_attenuation[0]
        max_ripple = max_ripple[0]
        sos = sp.signal.ellip(order, max_ripple, min_attenuation, cutoff, btype="highpass", fs=fs, output="sos")
        return sp.signal.sosfilt(sos, signal)
    
    FunctionLibrary["BandPassFilter_Butter"] = BandPassFilter_Butter
    FunctionLibrary["BandStopFilter_Butter"] = BandStopFilter_Butter
    FunctionLibrary["LowPassFilter_Butter"] = LowPassFilter_Butter
    FunctionLibrary["HighPassFilter_Butter"] = HighPassFilter_Butter

    FunctionLibrary["BandPassFilter_Cheby1"] = BandPassFilter_Cheby1
    FunctionLibrary["BandStopFilter_Cheby1"] = BandStopFilter_Cheby1
    FunctionLibrary["LowPassFilter_Cheby1"] = LowPassFilter_Cheby1
    FunctionLibrary["HighPassFilter_Cheby1"] = HighPassFilter_Cheby1

    FunctionLibrary["BandPassFilter_Cheby2"] = BandPassFilter_Cheby2
    FunctionLibrary["BandStopFilter_Cheby2"] = BandStopFilter_Cheby2
    FunctionLibrary["LowPassFilter_Cheby2"] = LowPassFilter_Cheby2
    FunctionLibrary["HighPassFilter_Cheby2"] = HighPassFilter_Cheby2

    FunctionLibrary["BandPassFilter_Ellip"] = BandPassFilter_Ellip
    FunctionLibrary["BandStopFilter_Ellip"] = BandStopFilter_Ellip
    FunctionLibrary["LowPassFilter_Ellip"] = LowPassFilter_Ellip
    FunctionLibrary["HighPassFilter_Ellip"] = HighPassFilter_Ellip

    FunctionLibrary["ClipSignal"] = ClipSignal

except ImportError:
    pass