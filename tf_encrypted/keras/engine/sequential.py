"""Sequential model API."""

from tf_encrypted.keras.engine.base_layer import Layer


class Sequential(Layer):
  """Model defined by a stack of layers in sequence.

  TODO
  """
  def __init__(self, layers=None, name=None):
    super(Sequential, self).__init__(name=name)

    # Add to the model any layers passed to the constructor.
    if layers:
      for layer in layers:
        self.add(layer)

  def add(self, layer):
    """Adds a layer instance on top of the layer stack.
    Arguments:
        layer: layer instance.
    Raises:
        TypeError: If `layer` is not a layer instance.
        ValueError: In case the `layer` argument does not
            know its input shape.
        ValueError: In case the `layer` argument has
            multiple output tensors, or is already connected
            somewhere else (forbidden in `Sequential` models).
    """
    if not isinstance(layer, Layer):
      raise TypeError('The added layer must be '
                      'an instance of class Layer. '
                      'Found: ' + str(layer))
    self.built = False
    set_inputs = False
    if not self._layers:
      if isinstance(layer, InputLayer):
        raise ValueError("Do not manually define an InputLayer in your "
                         "tfe.keras.Sequential model.")
      else:
        batch_shape = layer._batch_input_shape  # pylint: disable=protected-access

        # Instantiate an input layer.
        x = Input(
            batch_shape=batch_shape,
            name=layer.name + '_input')
        # This will build the current layer
        # and create the node connecting the current layer
        # to the input layer we just created.
        y = [layer(x)]

        # If an input layer (placeholder) is available.
        if isinstance(y, (tuple, list)):
          raise ValueError('All layers in a Sequential model '
                           'should have a single output tensor. '
                           'For multi-output layers, '
                           'use the functional API.')
        self.outputs = [y]

    elif self.outputs:
      # If the model is being built continuously on top of an input layer:
      # refresh its output.
      output_tensor = layer(self.outputs[0])
      if isinstance(output_tensor, list):
        raise TypeError('All layers in a Sequential model '
                        'should have a single output tensor. '
                        'For multi-output layers, '
                        'use the functional API.')
      self.outputs = [output_tensor]
    if set_inputs:
      self.built = True
    else:
      self._layers.append(layer)

  def pop(self): """Return last layer in a model."""

  @property
  def layers(self):
    """Historically, `sequential.layers` only returns layers that were added
    via `add`, and omits the auto-generated `InputLayer` that comes at the
    bottom of the stack."""
    layers = self._layers
    if layers and isinstance(layers[0], InputLayer):
      return layers[1:]
    return layers[:]
