"""
Author: Damien GUEHO
Copyright: Copyright (C) 2023 Damien GUEHO
License: Public Domain
Version: 24
"""




class continuous_signal:
    def __init__(self, **kwargs):
        self.signal_type = 'continuous'

        self.signal = kwargs.get('signal', lambda t: 0)
