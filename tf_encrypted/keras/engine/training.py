"""Functional Model API."""
from tf_encrypted.keras.engine.network import Network


class Model(Network):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
