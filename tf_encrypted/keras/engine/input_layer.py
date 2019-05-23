from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.base_layer import Node
from tensorflow.python.keras.utils import tf_utils

from tf_encrypted.keras.engine.base_layer import Layer

class InputLayer(Layer):
  """Layer to be used as an entry point into a Network (a graph of layers).
  It can either wrap an existing tensor (pass an `input_tensor` argument)
  or create its a placeholder tensor (pass arguments `input_shape`, and
  optionally, `dtype`).
  It is generally recommend to use the functional layer API via `Input`,
  (which creates an `InputLayer`) without directly using `InputLayer`.
  Arguments:
      input_shape: Shape tuple (not including the batch axis), or `TensorShape`
        instance (not including the batch axis).
      batch_size: Optional input batch size (integer or None).
      dtype: Datatype of the input.
      input_tensor: Optional tensor to use as layer input
          instead of creating a placeholder.
      sparse: Boolean, whether the placeholder created
          is meant to be sparse.
      name: Name of the layer (string).
  """

  def __init__(self,
               input_shape=None,
               batch_size=None,
               dtype=None,
               input_tensor=None,
               sparse=False,
               name=None,
               **kwargs):
    if 'batch_input_shape' in kwargs:
      batch_input_shape = kwargs.pop('batch_input_shape')
      if input_shape and batch_input_shape:
        raise ValueError('Only provide the input_shape OR '
                         'batch_input_shape argument to '
                         'InputLayer, not both at the same time.')
      batch_size = batch_input_shape[0]
      input_shape = batch_input_shape[1:]
    if kwargs:
      raise ValueError('Unrecognized keyword arguments:', kwargs.keys())

    if not name:
      prefix = 'input'
      name = prefix + '_' + str(backend.get_uid(prefix))

    if not dtype:
      if input_tensor is None:
        dtype = backend.floatx()
      else:
        dtype = backend.dtype(input_tensor)
    super(InputLayer, self).__init__(dtype=dtype, name=name)
    self.built = True
    self.sparse = sparse
    self.batch_size = batch_size
    self.supports_masking = True

    if isinstance(input_shape, tensor_shape.TensorShape):
      input_shape = tuple(input_shape.as_list())

    if input_tensor is None:
      if input_shape is not None:
        batch_input_shape = (batch_size,) + tuple(input_shape)
      else:
        batch_input_shape = None
      graph = backend.get_graph()
      with graph.as_default():
        # In graph mode, create a graph placeholder to call the layer on.
        if sparse:
          input_tensor = backend.placeholder(
              shape=batch_input_shape,
              dtype=dtype,
              name=self.name,
              sparse=True)
        else:
          input_tensor = backend.placeholder(
              shape=batch_input_shape,
              dtype=dtype,
              name=self.name)

      self.is_placeholder = True
      self._batch_input_shape = batch_input_shape
    else:
      if not tf_utils.is_symbolic_tensor(input_tensor):
        raise ValueError('You should not pass an EagerTensor to `Input`. '
                         'For example, instead of creating an '
                         'InputLayer, you should instantiate your model and '
                         'directly call it on your input.')
      self.is_placeholder = False
      self._batch_input_shape = tuple(input_tensor.get_shape().as_list())

    # Create an input node to add to self.outbound_node
    # and set output_tensors' _keras_history.
    input_tensor._keras_history = (self, 0, 0)  # pylint: disable=protected-access
    Node(
        self,
        inbound_layers=[],
        node_indices=[],
        tensor_indices=[],
        input_tensors=[input_tensor],
        output_tensors=[input_tensor])

  def get_config(self):
    config = {
        'batch_input_shape': self._batch_input_shape,
        'dtype': self.dtype,
        'sparse': self.sparse,
        'name': self.name
    }
    return config
