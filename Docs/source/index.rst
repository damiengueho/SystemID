Documentation for systemID
====================================

.. image:: images/logo/4570_4-3.png
  :align: center
  :width: 600
  :alt: Alternative text

**systemID** is an open source Python package consisting of a collection of functions useful for linear and nonlinear system identification, focusing on time-domain methods.
It provides a complete and robust API and handles most of the latest system identification techniques.

Specifically, it handles
    * Eigensystem Realization Algorithm (ERA)
    * Observer Kalman Identification Algorithm (OKID)
    * Eigensystem Realization Algorithm with Data Correlations (ERA/DC)
    * Time-varying Eigensystem Realization Algorithm (TVERA)
    * Time-varying Observer Kalman Identification Algorithm (TVOKID)
    * Bilinear System Identification Algorithm
    * Koopman Operator
    * Sparse Approximation



.. toctree::
   :hidden:
   :caption: Getting Started

   gettingStarted/requirements
   gettingStarted/installation
   gettingStarted/relatedPackages


.. toctree::
   :hidden:
   :caption: Documentation

   documentation/classes
   documentation/functions
   documentation/plotting


.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials/linearTimeInvariantDynamics
   tutorials/linearTimeVaryingDynamics
   tutorials/bilinearDynamics
   tutorials/nonlinearDynamics


.. toctree::
   :hidden:
   :caption: Background

   background/aboutSystemID
   background/problemsAndSuggestions
   background/acknowledgements
   background/references
