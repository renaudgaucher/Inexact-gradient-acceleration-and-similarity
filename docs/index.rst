.. Documentation for Inexact Gradient Acceleration and Similarity.

.. toctree::
  :hidden:
  :titlesonly:
  
  aggregators/index
  attacks/index
  fed_framework/index


Inexact Gradient Acceleration and Similarity
============================================

This research repository is derived from the
`ByzFL project <https://github.com/LPD-EPFL/byzfl>`_, with modifications for
inexact-gradient acceleration and similarity experiments. It is not the
official ByzFL project or its official documentation.

Project overview
================

The codebase studies Byzantine-resilient federated learning using robust
aggregation, Byzantine attacks, and accelerated or proximal training methods.
Its implementation is in the ``byzfl`` package and builds on the upstream
ByzFL research framework.

Key Features
------------

1. **Robust Aggregators and Pre-Aggregators**: Implement state-of-the-art aggregation strategies, such as :ref:`trmean-label`, alongside pre-aggregation techniques like :ref:`clipping-label` and :ref:`nnm-label`, to ensure resilience against Byzantine participants in federated learning workflows.
2. **Byzantine Attacks**: Simulate a wide range of adversarial attack scenarios, including :ref:`ipm-label` and :ref:`alie-label`, to rigorously test the robustness of federated learning systems under malicious conditions.
3. **Comprehensive Federated Learning Pipelines**: Seamlessly integrate key components — such as :ref:`client-label`, :ref:`server-label`, :ref:`byzantine-client-label`, and :ref:`data-dist-label` — to train, simulate, and benchmark robust aggregation schemes. The framework supports various datasets, models, and custom configurations, enabling researchers to replicate real-world distributed learning setups with heterogeneity and adversarial threats.


Installation
============

Install this repository in editable mode from its root:

.. code-block:: bash

   python -m pip install -e .


After installation, the package is ready to use. The example below applies the
:ref:`trmean-label` robust aggregator and the :ref:`sf-label` Byzantine attack:


Quick Start Example - Using PyTorch Tensors
-------------------------------------------

.. code-block:: python

  import byzfl
  import torch

  # Number of Byzantine participants
  f = 1

  # Honest vectors
  honest_vectors = torch.tensor([[1., 2., 3.],
                                 [4., 5., 6.],
                                 [7., 8., 9.]])

  # Initialize and apply the attack
  attack = byzfl.SignFlipping()
  byz_vector = attack(honest_vectors)

  # Create f identical attack vectors
  byz_vectors = byz_vector.repeat(f, 1)

  # Concatenate honest and Byzantine vectors
  all_vectors = torch.cat((honest_vectors, byz_vectors), dim=0)

  # Initialize and perform robust aggregation
  aggregate = byzfl.TrMean(f=f)
  result = aggregate(all_vectors)
  print("Aggregated result:", result)

Output:

.. code-block:: none

   Aggregated result: tensor([2.5000, 3.5000, 4.5000])


Quick Start Example - Using NumPy Arrays
----------------------------------------

.. code-block:: python

   import byzfl
   import numpy as np

   # Number of Byzantine participants
   f = 1

   # Honest vectors
   honest_vectors = np.array([[1., 2., 3.],
                              [4., 5., 6.],
                              [7., 8., 9.]])

   # Initialize and apply the attack
   attack = byzfl.SignFlipping()
   byz_vector = attack(honest_vectors)

   # Create f identical attack vectors
   byz_vectors = np.tile(byz_vector, (f, 1))

   # Concatenate honest and Byzantine vectors
   all_vectors = np.concatenate((honest_vectors, byz_vectors), axis=0)

   # Initialize and perform robust aggregation
   aggregate = byzfl.TrMean(f=f)
   result = aggregate(all_vectors)
   print("Aggregated result:", result)

Output:

.. code-block:: none

   Aggregated result: [2.5 3.5 4.5]

Learn More
==========

Explore the key components below:

.. raw:: html

  <div class="row">
  <div class="column">
    <div class="card">
      <div class="container">
        <h2>Aggregators</h2>
        <p>

:ref:`Get Started <aggregations-label>`

.. raw:: html

        </p>
      </div>
    </div>
  </div>

  <div class="column">
    <div class="card">
      <div class="container">
        <h2>Attacks</h2>

:ref:`Get Started <attacks-label>`

.. raw:: html

      </div>
    </div>
  </div>

  <div class="column">
    <div class="card">
      <div class="container">
        <h2>FL Framework</h2>

:ref:`Get Started <federated-learning-framework-label>`

.. raw:: html

      </div>
    </div>
  </div>
  </div>

Provenance and reuse
====================

This repository does not currently provide a general reuse license. Public
redistribution of this derivative was authorized by the upstream team, but
terms for reuse of the combined codebase have not been established. Do not
assume that public availability grants permission to reuse the code.

This repository is hosted on
`GitHub <https://github.com/renaudgaucher/Inexact-gradient-acceleration-and-similarity>`_.
For the upstream project, see
`LPD-EPFL/byzfl <https://github.com/LPD-EPFL/byzfl>`_.