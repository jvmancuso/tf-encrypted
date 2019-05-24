"""Sequential model API."""

from tf_encrypted.keras.engine.training import Model


class Sequential(Model):
  """Model defined by a stack of layers in sequence.
  
  TODO
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def add(self): """Add a layer."""

  def pop(self): """Return last layer in a model."""
