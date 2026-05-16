Instruments Module
==================

This subpackage provides parsers for reading experimental instrument export
files commonly used in fire science and fire safety engineering research.
Supported formats include thermal analysis (STA, DSC, TGA) and calorimetry
(MCC, Cone Calorimeter) instruments.

A generic reader with automatic file type detection is available via
:func:`firescipy.instruments.read_instrument_file`.


Reader
------

.. automodule:: firescipy.instruments.reader
   :members:
   :undoc-members:
   :show-inheritance:


Netzsch STA / DSC / TGA
------------------------

.. automodule:: firescipy.instruments.netzsch_sta
   :members:
   :undoc-members:
   :show-inheritance:


DEATAK MCC
----------

.. automodule:: firescipy.instruments.deatak_mcc
   :members:
   :undoc-members:
   :show-inheritance:


Netzsch Cone Calorimeter
------------------------

.. automodule:: firescipy.instruments.netzsch_cone
   :members:
   :undoc-members:
   :show-inheritance:
