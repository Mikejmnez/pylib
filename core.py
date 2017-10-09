"""
Core functions for pylib
"""
import numpy as np
import warnings

def power_spectral_density(time, signal, window_length=-1, overlap=0, \
    window='rectangular'):
    """
    Estimates the power spectral density of "signal" measured at "time." Non-
    uniformly spaced signals will be linearly interpolated onto an equally
    spaced array with the same number of elements as the original signal.

    By default, the estimate will be computed for the entire signal with a
    rectangular windowing function.

    "window_length" is the length of the windows used the perform the estimate
    in the same units as "time". A negative value will use the entire signal as
    the window.

    "overlap" is the fraction of the window that overlaps the next window. This
    is rounded down to the nearest integer of the step size in "time"

    "window" is the windowing function used. Allowable values are: rectangular,
    bartlett, blackman, hanning, and hamming.

    """
    # check inputs
    assert np.size(time) == np.size(signal), "time and signal must have the same length"

    # interpolate onto a regularly spaced time and subtract mean
    t = np.linspace(time[0],time[-1],np.size(time))
    v = np.interp(t,time,signal)
    v = v - np.mean(v)
    dt = (t[-1]-t[0])/np.size(t)

    # if window_length is negative, use only 1 window
    if (window_length < 0):
        window_length = t[-1] + dt

    # reduce window and overlap sizes to an integer multiple of the time step
    # make sure the overlap is not the same as the window (infinite windows)
    # determine number of windows
    M = int(window_length // dt)         # number of points in window
    D = int(np.floor(M*overlap))         # number of overlapping points
    assert M > D, "there must be non-overlapping points in each window"
    N = int(np.size(t) // (M-D))         # number of windows

    # correct number of windows to keep from running off end of the array
    No = np.size(t) - ((M-D)*(N-1)+M)
    if No < 0:
        N = N - int( np.ceil(np.abs(No)/(M-D)) )

    # select the window
    if window == 'rectangular':
        W = np.ones(M)
    elif window == 'bartlett':
        W = np.bartlett(M)
    elif window == 'blackman':
        W = np.blackman(M)
    elif window == 'hanning':
        W = np.hanning(M)
    elif window == 'hamming':
        W = np.hamming(M)
    else:
        warnings.warn("Invalid window type. Using rectangular window.");
        W = np.ones(M)

    # calculate frequencies and preallocate power
    f = np.fft.rfftfreq(M, dt)                  # frequencies
    if np.mod(M,2):
        Nfft = (M+1)//2                          # odd
    else:
        Nfft = M//2+1                            # even
    P = np.zeros(Nfft)

    # iterate through each window
    for i in range(0, N):
        P = P + np.abs(np.fft.rfft(np.multiply(W, v[(M-D)*i:(M-D)*i+M])))/N

    # return frequency and power
    return [f, P]

def spectral_cutoff_filter(time, signal, low=0.0, high=float("inf")):
    """
    Applies a spectral cutoff filter of "signal" measured at "time." All
    frequencies outside the cutoffs "low" and "high", except the mean, are set
    to zero. The cutoffs are specified in inverse units of "time".

    In order to apply the filter, non-uniformly spaced signals will be linearly
    interpolated onto an equally paced array with the same number of elements as
    the original signal before applying the filter. The result is interpolated
    back onto the original time array.
    """
    # check inputs
    assert np.size(time) == np.size(signal), "time and signal must have the same length"
    assert low >= 0 and high >= 0, "cutoffs must be positive"
    assert low < high, " low must be smaller than high"

    # interpolate onto a regularly spaced time and subtract mean
    t = np.linspace(time[0], time[-1], np.size(time))
    v = np.interp(t,time,signal)
    dt = (t[-1]-t[0])/np.size(t)

    # apply filter
    N = np.size(v)                              # size of signal
    f = np.fft.rfftfreq(N, dt)                  # frequencies
    vbar = np.mean(v)
    vhat = np.fft.rfft(v-vbar)
    vhat[f < low] = 0
    vhat[f > high] = 0
    vf = np.fft.irfft(vhat)+vbar

    # interpolate back onto original time
    signal_filtered = np.interp(time,t,vf)
    return signal_filtered

def moving_average(time, signal, window):
    """
    Calculates a moving aveage of "signal" over a time-window of "window." The
    average is given for the window before each point of "time." For points
    without a full window available, the average is conmputed over the known
    values.
    """
    # Check inputs
    assert window > 0, "windowing length must be positive."
    assert np.size(time) == np.size(signal), "time and signal must have the same length"

    # calculate size of signal and interpolate onto uniform time
    N = np.size(signal)
    t = np.linspace(time[0], time[-1], np.size(time))
    v = np.interp(t,time,signal)
    dt = (t[-1]-t[0])/np.size(t)

    # round window to an integer multiple of the time step
    M = int(np.round(window/dt))

    # perform moving average
    cumsum = np.cumsum(v)
    avg = np.empty(N)
    avg[M:] = (cumsum[M:] - cumsum[:-M])/(M)
    for i in range(0, M):
        avg[i] = cumsum[i]/(i+1)

    # interpolate back to original signal
    avg = np.interp(time, t, avg)

    return avg

def softplus(x):
    """
    Evaluates the softplus function ln(1 + e^x)
    """
    return np.log(1.0 + np.exp(x))

class TDMA:
    """
    A class to solve a tridiagonal matrix system Ax=d using the tridiagonal
    matrix algorithm. A is a matrix of the form:
          --                                    --
          | b0      c0                           |
          | a1      b1      c1                   |
    A =   |         ...     ...     ...          |
          |                 an-2    bn-2    cn-2 |
          |                         an-1    bn-1 |
          --                                    --
    On construction, the vectors a, b, and c must have the same length. a must
    have 0 in its first element, and c must have 0 in its last element.
    """
    def __init__(self, a, b, c):
        # Check inputs
        assert np.size(b) == np.size(a), "b must be the same size as a"
        assert np.size(c) == np.size(a), "c must be the same size as a"
        assert a[0] == 0.0, "a's first element must contain 0"
        assert c[-1] == 0.0, "c's last element must contain 0"

        # Assign matrices to object
        self.n = np.size(a)
        self.a = a
        self.b = b
        self.c = c
        self.cp = np.zeros(np.shape(a))

        # Do a forward sweep
        self.cp[0] = c[0] / b[0]
        for i in range(1, self.n):
            self.cp[i] = c[i] / (b[i] - a[i]*self.cp[i-1])

    def solve(self, d):
        # check inputs
        assert np.size(d) == self.n, "d must have size %i" % self.n

        # allocate array
        x = np.array(d)
        dp = np.array(d)

        # Forward sweep
        dp[0] = d[0] / self.b[0]
        for i in range(1, self.n):
            dp[i] = (d[i] - self.a[i]*dp[i-1]) / (self.b[i] - self.a[i]*self.cp[i-1])

        # Backward sweep
        x[-1] = dp[-1]
        for i in range(self.n-2,-1,-1):
            x[i] = dp[i] - self.cp[i]*x[i+1]

        return x
