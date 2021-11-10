Installation
============

Development version
-------------------

To install the latest development version of ULaMDyn, clone the whole repository with

.. code-block:: console

	$ git clone https://gitlab.com/maxjr82/ulamdyn.git
	$ cd ulamdyn

Alternatively, you can just update your existing local copy with the command below:

.. code-block:: console

	$ git pull origin master

To proceed with the local installation, one should run the following pip command in terminal:

.. code-block:: console
        $ pip install -e .

.. note:: To install the package in the current users's home directory instead of system-wide (i.e., without root permissions), you should add the flag ``--user`` 
          to the pip command provided above. This option might require you to update your system's ``PATH`` variable accordingly.

The following external libraries will be automatically installed with ulamdyn:

   - numpy
   - scipy
   - pandas
   - scikit-learn
   - joblib
   - h5py
   - rmsd

Optional dependencies
---------------------

For handling large datasets, ULaMDyn also provides an interface to the `Modin <https://modin.readthedocs.io/en/latest/>`_ library which can be used to parallelize and
speed up operations on pandas DataFrame objects. Modin package is not installed by default with ULaMDyn. To enable its usage, one can add the optional dependency (or
"extra package") by specifying the package name during installation using the "square bracket syntax":

.. code-block:: console
        $ pip install ulamdyn[modin]

