"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""


from systemID.SystemIDAlgorithms.TimeVaryingObserverKalmanIdentificationAlgorithmWithObserver import timeVaryingObserverKalmanIdentificationAlgorithmWithObserver



class tvokid:
    def __init__(self, input_signals, output_signals, **kwargs):
        self.D, self.hki, self.hkio, self.hki_observer1, self.hki_observer2 = timeVaryingObserverKalmanIdentificationAlgorithmWithObserver(input_signals, output_signals, **kwargs)
