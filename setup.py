from setuptools import setup, find_packages

VERSION = '0.0.10'
DESCRIPTION = 'Package for time domain system identification. LTI and LTV systems, bilinear systems and nonlinear systems.'
LONG_DESCRIPTION = 'Package for time domain system identification. Supports linear time-invariant (LTI) and linear time-varying (LTV) dynamics, bilinear dynamics and nonlinear dynamics.'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="systemID",
    version=VERSION,
    author="Damien Gueho",
    author_email="<systemidtechnologies@gmail.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=[
        "systemID",
        "systemID.ClassesDynamics",
        "systemID.ClassesGeneral",
        "systemID.ClassesSparseID",
        "systemID.ClassesSystemID",
        "systemID.Plotting",
        "systemID.SparseIDAlgorithms",
        "systemID.SystemIDAlgorithms",
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
