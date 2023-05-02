from setuptools import setup, find_packages

VERSION = '24.22.1'
DESCRIPTION = 'Package for time domain system identification. LTI and LTV systems, bilinear systems and nonlinear systems.'
LONG_DESCRIPTION = 'Package for time domain system identification. Supports linear time-invariant (LTI) and linear time-varying (LTV) dynamics, bilinear dynamics and nonlinear dynamics.'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="systemID",
    version=VERSION,
    author="Damien Gueho",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=[
        "systemID",
        "systemID.signals",
        "systemID.state_space_model",
        "systemID.input_output_model",
        "systemID.dynamics",
        "systemID.state_space_identification",
        "systemID.input_output_identification",
        "systemID.functions",
        "systemID.ClassesDynamics",
        "systemID.ClassesGeneral",
        "systemID.ClassesSparseID",
        "systemID.ClassesSystemID",
        "systemID.Plotting",
        "systemID.SparseIDAlgorithms",
        "systemID.SystemIDAlgorithms"
    ],
    install_requires=['numpy', 'scipy', 'matplotlib'],
    keywords=['python', 'system identification'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
