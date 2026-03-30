Data: ``sj.data``
=================

The data module provides the core ``spatioloji`` container, configuration classes,
quality control, image handling, and export utilities.

Core classes
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.data.spatioloji
   spatioloji_s.data.SpatiolojiConfig
   spatioloji_s.data.SpatialData
   spatioloji_s.data.ExpressionMatrix

Image handling
--------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.data.ImageHandler
   spatioloji_s.data.ImageMetadata
   spatioloji_s.data.load_fov_positions_from_images

Quality control
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.data.spatioloji_qc
   spatioloji_s.data.xenium_qc
   spatioloji_s.data.QCConfig
   spatioloji_s.data.XeniumQCConfig

Export
------

.. autosummary::
   :toctree: generated
   :nosignatures:

   spatioloji_s.data.export_to_csv_bundle
