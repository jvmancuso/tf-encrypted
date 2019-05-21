"""Keras Network abstraction."""
from tf_encrypted.keras.engine.base_layer import Layer, Node

class Network(Layer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def call(self, inputs): """Forward pass."""
