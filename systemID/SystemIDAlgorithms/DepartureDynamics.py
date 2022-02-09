"""
Author: Damien GUEHO
Copyright: Copyright (C) 2021 Damien GUEHO
License: Public Domain
Version: 22
Date: February 2022
Python: 3.7.7
"""



import copy


from systemID.ClassesGeneral.ClassSignal import DiscreteSignal, ContinuousSignal, OutputSignal, subtract2Signals
from systemID.ClassesGeneral.ClassSystem import ContinuousNonlinearSystem, DiscreteNonlinearSystem
from systemID.ClassesGeneral.ClassExperiments import Experiments


def departureDynamics(nominal_system, nominal_input_signal, tspan, deviations_dx0, deviations_input_signal, full_deviation_dx0, full_deviation_input_signal):

    # Dimensions
    state_dimension = nominal_system.state_dimension
    input_dimension = nominal_system.input_dimension
    output_dimension = nominal_system.output_dimension


    # Time
    total_time = tspan[-1]
    frequency = int(round((len(tspan) - 1) / total_time))


    # Number of experiments
    number_free_decay_experiments = len(deviations_dx0)
    number_forced_response_experiments = len(deviations_input_signal)


    # Integration Nominal Trajectory
    nominal_output_signal = OutputSignal(nominal_input_signal, nominal_system, tspan=tspan)


    # Create Free Decay Experiments
    free_decay_input_signals = []
    free_decay_systems = []
    for i in range(number_free_decay_experiments):
        initial_state = [(deviations_dx0[i] + nominal_system.x0, 0)]
        free_decay_systems.append(ContinuousNonlinearSystem(state_dimension, input_dimension, output_dimension, initial_state, 'Free Decay Experiment System' + str(i), nominal_system.F, nominal_system.G))
        free_decay_input_signals.append(ContinuousSignal(input_dimension, signal_shape='External', u=nominal_input_signal.u))
    free_decay_experiments = Experiments(free_decay_systems, free_decay_input_signals, tspan=tspan)
    free_decay_experiments_deviated = copy.deepcopy(free_decay_experiments)
    for i in range(number_free_decay_experiments):
        free_decay_experiments_deviated.input_signals[i] = DiscreteSignal(input_dimension, total_time, frequency)
        free_decay_experiments_deviated.output_signals[i] = subtract2Signals(free_decay_experiments.output_signals[i], nominal_output_signal)


    # Create Forced Response Experiments
    forced_response_input_signals = []
    forced_response_systems = []
    for i in range(number_forced_response_experiments):
        forced_response_systems.append(ContinuousNonlinearSystem(state_dimension, input_dimension, output_dimension, nominal_system.initial_states, 'Forced Response Experiment System' + str(i), nominal_system.F, nominal_system.G))
        def make_u(i):
            def u(t):
                return nominal_input_signal.u(t) + deviations_input_signal[i].u(t)
            return u
        forced_response_input_signals.append(ContinuousSignal(input_dimension, signal_shape='External', u=make_u(i)))
    forced_response_experiments = Experiments(forced_response_systems, forced_response_input_signals, tspan=tspan)
    forced_response_experiments_deviated = copy.deepcopy(forced_response_experiments)
    for i in range(number_forced_response_experiments):
        data = deviations_input_signal[i].u(tspan)
        forced_response_experiments_deviated.input_signals[i] = DiscreteSignal(input_dimension, total_time, frequency, signal_shape='External', data=data)
        forced_response_experiments_deviated.output_signals[i] = subtract2Signals(forced_response_experiments.output_signals[i], nominal_output_signal)


    # Create Full Experiment
    full_system = ContinuousNonlinearSystem(state_dimension, input_dimension, output_dimension, [(full_deviation_dx0 + nominal_system.x0, 0)], 'Full Experiment System', nominal_system.F, nominal_system.G)
    def u(t):
        return nominal_input_signal.u(t) + full_deviation_input_signal.u(t)
    full_input_signal = ContinuousSignal(input_dimension, signal_shape='External', u=u)
    full_experiment = Experiments([full_system], [full_input_signal], tspan=tspan)
    full_experiment_deviated = copy.deepcopy(full_experiment)
    data = full_deviation_input_signal.u(tspan)
    full_experiment_deviated.input_signals[0] = DiscreteSignal(input_dimension, total_time, frequency, signal_shape='External', data=data)
    full_experiment_deviated.output_signals[0] = subtract2Signals(full_experiment.output_signals[0], nominal_output_signal)


    return free_decay_experiments, free_decay_experiments_deviated, forced_response_experiments, forced_response_experiments_deviated, full_experiment, full_experiment_deviated






















def departureDynamicsDiscrete(nominal_system, nominal_input_signal, tspan, deviations_dx0, deviations_input_signal, full_deviation_dx0, full_deviation_input_signal):

    # Dimensions
    state_dimension = nominal_system.state_dimension
    input_dimension = nominal_system.input_dimension
    output_dimension = nominal_system.output_dimension


    # Time
    total_time = tspan[-1]
    frequency = int(round((len(tspan) - 1) / total_time))


    # Number of experiments
    number_free_decay_experiments = len(deviations_dx0)
    number_forced_response_experiments = len(deviations_input_signal)


    # Integration Nominal Trajectory
    nominal_output_signal = OutputSignal(nominal_input_signal, nominal_system, tspan=tspan)


    # Create Free Decay Experiments
    free_decay_input_signals = []
    free_decay_systems = []
    for i in range(number_free_decay_experiments):
        initial_state = [(deviations_dx0[i] + nominal_system.x0, 0)]
        free_decay_systems.append(DiscreteNonlinearSystem(frequency, state_dimension, input_dimension, output_dimension, initial_state, nominal_system.F, nominal_system.G))
        free_decay_input_signals.append(DiscreteSignal(input_dimension, total_time, frequency, signal_shape='External', data=nominal_input_signal.data))
    free_decay_experiments = Experiments(free_decay_systems, free_decay_input_signals, tspan=tspan)
    free_decay_experiments_deviated = copy.deepcopy(free_decay_experiments)
    for i in range(number_free_decay_experiments):
        free_decay_experiments_deviated.input_signals[i] = DiscreteSignal(input_dimension, total_time, frequency)
        free_decay_experiments_deviated.output_signals[i] = subtract2Signals(free_decay_experiments.output_signals[i], nominal_output_signal)


    # Create Forced Response Experiments
    forced_response_input_signals = []
    forced_response_systems = []
    for i in range(number_forced_response_experiments):
        forced_response_systems.append(DiscreteNonlinearSystem(frequency, state_dimension, input_dimension, output_dimension, nominal_system.initial_states, 'Forced Response Experiment System' + str(i), nominal_system.F, nominal_system.G))
        # def make_u(i):
        #     def u(t):
        #         return nominal_input_signal.u(t) + deviations_input_signal[i].u(t)
        #     return u

        forced_response_input_signals.append(DiscreteSignal(input_dimension, total_time, frequency, signal_shape='External', data=nominal_input_signal.data + deviations_input_signal[i].u(tspan)))
    forced_response_experiments = Experiments(forced_response_systems, forced_response_input_signals, tspan=tspan)
    forced_response_experiments_deviated = copy.deepcopy(forced_response_experiments)
    for i in range(number_forced_response_experiments):
        data = deviations_input_signal[i].u(tspan)
        forced_response_experiments_deviated.input_signals[i] = DiscreteSignal(input_dimension, total_time, frequency, signal_shape='External', data=data)
        forced_response_experiments_deviated.output_signals[i] = subtract2Signals(forced_response_experiments.output_signals[i], nominal_output_signal)


    # Create Full Experiment
    full_system = DiscreteNonlinearSystem(frequency, state_dimension, input_dimension, output_dimension, [(full_deviation_dx0 + nominal_system.x0, 0)], 'Full Experiment System', nominal_system.F, nominal_system.G)
    # def u(t):
    #     return nominal_input_signal.u(t) + full_deviation_input_signal.u(t)
    full_input_signal = DiscreteSignal(input_dimension, 'Full Experiment Input Signal', total_time, frequency, signal_shape='External', data=nominal_input_signal.data + full_deviation_input_signal.u(tspan))
    full_experiment = Experiments([full_system], [full_input_signal], tspan=tspan)
    full_experiment_deviated = copy.deepcopy(full_experiment)
    data = full_deviation_input_signal.u(tspan)
    full_experiment_deviated.input_signals[0] = DiscreteSignal(input_dimension, total_time, frequency, signal_shape='External', data=data)
    full_experiment_deviated.output_signals[0] = subtract2Signals(full_experiment.output_signals[0], nominal_output_signal)


    return free_decay_experiments, free_decay_experiments_deviated, forced_response_experiments, forced_response_experiments_deviated, full_experiment, full_experiment_deviated























def departureDynamicsFromInitialConditionResponse(nominal_system, tspan, deviations_dx0):
    """
    Purpose:


    Parameters:
        -

    Returns:
        -

    Imports:
        -

    Description:


    See Also:
        -
    """

    # Dimensions
    state_dimension = nominal_system.state_dimension
    input_dimension = nominal_system.input_dimension
    output_dimension = nominal_system.output_dimension


    # Time
    total_time = tspan[-1]
    frequency = (len(tspan) - 1) / total_time


    # Number of experiments
    number_free_decay_experiments = len(deviations_dx0)


    # Integration Nominal Trajectory
    nominal_output_signal = OutputSignal(ContinuousSignal(input_dimension), nominal_system, tspan=tspan)


    # Create Free Decay Experiments
    free_decay_input_signals = []
    free_decay_systems = []
    for i in range(number_free_decay_experiments):
        initial_state = [(deviations_dx0[i] + nominal_system.x0, 0)]
        free_decay_systems.append(ContinuousNonlinearSystem(state_dimension, input_dimension, output_dimension, initial_state, 'Free Decay Experiment System' + str(i), nominal_system.F, nominal_system.G))
        free_decay_input_signals.append(ContinuousSignal(input_dimension))
    free_decay_experiments = Experiments(free_decay_systems, free_decay_input_signals, tspan=tspan)
    free_decay_experiments_deviated = copy.deepcopy(free_decay_experiments)
    for i in range(number_free_decay_experiments):
        free_decay_experiments_deviated.input_signals[i] = DiscreteSignal(input_dimension, total_time, frequency)
        free_decay_experiments_deviated.output_signals[i] = subtract2Signals(free_decay_experiments.output_signals[i], nominal_output_signal)

    # ## Create Full Experiment
    # full_system = ContinuousNonlinearSystem(state_dimension, input_dimension, output_dimension, [(full_deviation_dx0 + nominal_system.x0, 0)], 'Full Experiment System', nominal_system.F, nominal_system.G)
    # full_input_signal = ContinuousSignal(input_dimension)
    # full_experiment = Experiments([full_system], [full_input_signal], tspan=tspan)
    # full_experiment_deviated = copy.deepcopy(full_experiment)
    # full_experiment_deviated.input_signals[0] = DiscreteSignal(input_dimension, total_time, frequency)
    # full_experiment_deviated.output_signals[0] = subtract2Signals(full_experiment.output_signals[0], nominal_output_signal)


    return free_decay_experiments, free_decay_experiments_deviated
