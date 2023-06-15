import numpy as np
import scipy.signal as sps
import biosppy.signals as sbvp



def preprocessing(eda, cardio, temp, cardio_sampling_rate=700, bvp=True, label=None, init_time=0):
    """
    Pre-process physiological emotional response
    Argument:
        eda: numpy.array
            EDA signal
        
        cardio: numpy.array
            ECG or BVP signal
        
        temp: numpy.array
            Skin temperature signal
        
        cardio_sampling_rate: integer, optional
            Sampling rate of the cardio signal (default is 700)
        
        bvp: bool, optional
            Indicates if the cardio signal is BVP or ECG (default is True indicating BVP)
        
        label: numpy.array, optinal
            If labels are available. (default is None)
        
        init_time: integer, UNIX format
            The time in UNIX format of the begining of the recording (default is 0)
        
        Returns:
            signal: numpy.array
                The preprocessed signals. dimension 1: the obsarvation. dimension 2: the pre-processed signals eda, bvp, temp
        
            time: numpy.array
                The preprocessed time of measurement of each pre-processed observation
        
            label_true: numpy.array
                True labels resampled to match the length of *signal*. Only provided if *label* is not None.
    """
    

    ### Pre-processing
    if bvp:
        time, filtered_cardio, rpacks, _, _ = sbvp.bvp.bvp(signal=cardio, sampling_rate=cardio_sampling_rate, show=False)
    else:
        time, filtered_cardio, rpacks, _, _, _, _ = sbvp.ecg.ecg(signal=cardio, sampling_rate=cardio_sampling_rate, show=False)

    
    cardio, time = sps.resample(cardio, cardio.size//1, t=time)


    time += init_time


    ### Savgol filter
    window_length = min(151, eda.size -1)
    window_length += (window_length -1)%2
    eda = sps.savgol_filter(eda, window_length=window_length, polyorder=1)

    window_length = min(151, temp.size -1)
    window_length += (window_length -1)%2
    temp = sps.savgol_filter(temp, window_length=window_length, polyorder=1)

    ### Downsampling the labels and the biosignals
    eda = sps.resample(eda, cardio.size)
    temp = sps.resample(temp, cardio.size)

    if label is not None:
        change = label[1:] -label[:-1]
        mask = change != 0
        change_points = np.abs(change[mask])
        mask = np.concatenate((np.array([False]), mask))
        time_change = time[mask]
        
        label_true = np.zeros(time.size, dtype=int)
        time_change_size = time_change.size
        time_change_size -= time_change_size % 2
        for i in range(0, time_change_size, 2):
            mask = np.logical_and(time_change[i] < time, time < time_change[i+1])
            label_true[mask] = change_points[i]

    ### Rescaling
    eda = (eda -eda.min())/(eda.max() -eda.min())
    cardio = (cardio -cardio.min())/(cardio.max() -cardio.min())
    temp = (temp -temp.min())/(temp.max() -temp.min())
    
    ### Concatenation
    signal = np.stack((eda, cardio, temp), axis=1)

    if label is not None:
        return signal, time, label_true
    else:
        return signal, time
