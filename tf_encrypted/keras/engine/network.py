"""Keras Network abstraction."""
from tf_encrypted.keras.engine.base_layer import Layer, Node

class Network(Layer):
  """A `Network` is a composition of layers.

  `Network` is the topological form of a "model". A `Model`
  is simply a `Network` with added training routines.

  Two types of `Networks` exist: Graph Networks and Subclass Networks. Graph
  networks are used in the Keras Functional and Sequential APIs. Subclassed
  networks are used when a user subclasses the `Model` class. In general,
  more Keras features are supported with Graph Networks than with Subclassed
  Networks, specifically:

  - Model cloning (`keras.models.clone`)
  - Serialization (`model.get_config()/from_config`, `model.to_json()/to_yaml()`
  - Whole-model saving (`model.save()`)

  A Graph Network can be instantiated by passing two arguments to `__init__`.
  The first argument is the `keras.Input` Tensors that represent the inputs
  to the Network. The second argument specifies the output Tensors that
  represent the outputs of this Network. Both arguments can be a nested
  structure of Tensors.

  Example:

  ```
  inputs = {'x1': keras.Input(shape=(10,)), 'x2': keras.Input(shape=(1,))}
  t = keras.layers.Dense(1, activation='relu')(inputs['x1'])
  outputs = keras.layers.Add()([t, inputs['x2'])
  network = Network(inputs, outputs)
  ```

  A Graph Network constructed using the Functional API can also include raw
  TensorFlow functions, with the exception of functions that create Variables
  or assign ops.

  Example:

  ```
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(1)(inputs)
  outputs = tf.nn.relu(x)
  network = Network(inputs, outputs)
  ```

  Subclassed Networks can be instantiated via `name` and (optional) `dynamic`
  keyword arguments. Subclassed Networks keep track of their Layers, and their
  `call` method can be overridden. Subclassed Networks are typically created
  indirectly, by subclassing the `Model` class.

  Example:

  ```
  class MyModel(keras.Model):
    def __init__(self):
      super(MyModel, self).__init__(name='my_model', dynamic=False)

      self.layer1 = keras.layers.Dense(10, activation='relu')

    def call(self, inputs):
      return self.layer1(inputs)
  ```

  Allowed args in `super().__init__`:
    name: String name of the model.
    dynamic: (Subclassed models only) Set this to `True` if your model should
      only be run eagerly, and should not be used to generate a static
      computation graph. This attribute is automatically set for Functional API
      models.
    trainable: Boolean, whether the model's variables should be trainable.
    dtype: (Subclassed models only) Default dtype of the model's weights (
      default of `None` means use the type of the first input). This attribute
      has no effect on Functional API models, which do not have weights of their
      own.
  """
  def __init__(self, *args, **kwargs):  # pylint: disable=super-init-not-called
    # Signature detection
    if (len(args) == 2 or
        len(args) == 1 and 'outputs' in kwargs or
        'inputs' in kwargs and 'outputs' in kwargs):
      # Graph network
      self._init_graph_network(*args, **kwargs)
    else:
      # Subclassed network
      self._init_subclassed_network(**kwargs)

  def _base_init(self, name=None, **kwargs):
    # The following are implemented as property functions:
    # self.trainable_weights
    # self.non_trainable_weights
    # self.input_spec
    # self.losses
    # self.updates

    generic_utils.validate_kwargs(kwargs, {'trainable', 'dtype'})

    self._init_set_name(name, zero_based=True)
    self._activity_regularizer = None
    # This acts just like the `trainable` attribute of any layer instance.
    self._trainable = kwargs.get('trainable', True)
    self._is_compiled = False
    self._expects_training_arg = False
    self._layers = []

    # This is True for Sequential networks and Functional networks.
    self._compute_output_and_mask_jointly = False

    self.supports_masking = False
    if not hasattr(self, 'optimizer'):
      # Don't reset optimizer if already set.
      self.optimizer = None

    # Private attributes to implement compatibility with Layer.
    self._trainable_weights = []
    self._non_trainable_weights = []
    self._updates = []  # Used in symbolic mode only.
    self._losses = []
    self._eager_losses = []
    self._callable_losses = []
    # A list of metric instances corresponding to the symbolic metric tensors
    # added using the `add_metric` API.
    self._metrics = []
    # A dictionary that maps metric names to metric result tensors.
    self._metrics_tensors = {}
    self._scope = None  # Never used.
    self._reuse = None  # Never used.
    if context.executing_eagerly():
      self._graph = None
    else:
      self._graph = ops.get_default_graph()  # Used in symbolic mode only.
      # A Network does not create weights of its own, thus has no dtype.
    self._dtype = kwargs.get('dtype', None)

    # All layers in order of horizontal graph traversal.
    # Entries are unique. Includes input and output layers.
    self._layers = []

    # Used in symbolic mode only, only in conjunction with graph-networks
    self._outbound_nodes = []
    self._inbound_nodes = []

  def _init_graph_network(self, inputs, outputs, name=None, **kwargs):
    generic_utils.validate_kwargs(
        kwargs, {'trainable'},
        'Functional models may only specify `name` and `trainable` keyword '
        'arguments during initialization. Got an unexpected argument:')

    # Normalize and set self.inputs, self.outputs.
    if isinstance(inputs, list) and len(nest.flatten(inputs)) == 1:
      inputs = inputs[0]
    if isinstance(outputs, list) and len(nest.flatten(outputs)) == 1:
      outputs = outputs[0]
    self._nested_outputs = outputs
    self._nested_inputs = inputs
    self.inputs = nest.flatten(inputs)
    self.outputs = nest.flatten(outputs)

    if any(not hasattr(tensor, '_keras_history') for tensor in self.outputs):
      base_layer_utils.create_keras_history(self._nested_outputs)

    self._base_init(name=name, **kwargs)
    self._validate_graph_inputs_and_outputs()

    # A Network does not create weights of its own, thus it is already
    # built.
    self.built = True
    self._compute_output_and_mask_jointly = True
    self._is_graph_network = True
    # `_expects_training_arg` is True since the `training` argument is always
    # present in the signature of the `call` method of a graph network.
    self._expects_training_arg = True

    self._input_layers = []
    self._output_layers = []
    self._input_coordinates = []
    self._output_coordinates = []

    # This is for performance optimization when calling the Network on new
    # inputs. Every time the Network is called on a set on input tensors,
    # we compute the output tensors, output masks and output shapes in one pass,
    # then cache them here. When any of these outputs is queried later, we
    # retrieve it from there instead of recomputing it.
    self._output_mask_cache = {}
    self._output_tensor_cache = {}
    self._output_shape_cache = {}

    # Build self._output_layers:
    for x in self.outputs:
      layer, node_index, tensor_index = x._keras_history  # pylint: disable=protected-access
      self._output_layers.append(layer)
      self._output_coordinates.append((layer, node_index, tensor_index))

    # Build self._input_layers:
    for x in self.inputs:
      layer, node_index, tensor_index = x._keras_history  # pylint: disable=protected-access
      # It's supposed to be an input layer, so only one node
      # and one tensor output.
      assert node_index == 0
      assert tensor_index == 0
      self._input_layers.append(layer)
      self._input_coordinates.append((layer, node_index, tensor_index))

    # Keep track of the network's nodes and layers.
    nodes, nodes_by_depth, layers, layers_by_depth = _map_graph_network(
        self.inputs, self.outputs)
    self._network_nodes = nodes
    self._nodes_by_depth = nodes_by_depth
    self._layers = layers
    self._layers_by_depth = layers_by_depth
    self._layer_call_argspecs = {}
    for layer in self._layers:
      self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)

    self._track_layers(layers)

    # Create the node linking internal inputs to internal outputs.
    base_layer.Node(
        outbound_layer=self,
        inbound_layers=[],
        node_indices=[],
        tensor_indices=[],
        input_tensors=self._nested_inputs,
        output_tensors=self._nested_outputs)

    # Build self.input_names and self.output_names.
    self._set_output_names()
    self.input_names = []
    self._feed_input_names = []
    self._feed_inputs = []
    self._feed_input_shapes = []
    for i, layer in enumerate(self._input_layers):
      self.input_names.append(layer.name)
      if layer.is_placeholder:
        self._feed_input_names.append(layer.name)
        self._feed_input_shapes.append(backend.int_shape(self.inputs[i]))
        self._feed_inputs.append(layer.input)

  def _init_subclassed_network(self, name=None, **kwargs):
    raise NotImplementedError("Subclassed Networks not yet supported.")

  def _validate_graph_inputs_and_outputs(self):
    """Validates the inputs and outputs of a Graph Network."""
    # Check for redundancy in inputs.
    if len(set(self.inputs)) != len(self.inputs):
      raise ValueError('The list of inputs passed to the model '
                       'is redundant. '
                       'All inputs should only appear once.'
                       ' Found: ' + str(self.inputs))

    for x in self.inputs:
      # Check that x has appropriate `_keras_history` metadata.
      if not hasattr(x, '_keras_history'):
        cls_name = self.__class__.__name__
        raise ValueError('Input tensors to a ' + cls_name + ' ' +
                         'must come from `tf.keras.Input`. '
                         'Received: ' + str(x) +
                         ' (missing previous layer metadata).')
      # Check that x is an input tensor.
      # pylint: disable=protected-access
      layer = x._keras_history.layer
      if len(layer._inbound_nodes) > 1 or (
          layer._inbound_nodes and layer._inbound_nodes[0].inbound_layers):
        cls_name = self.__class__.__name__
        logging.warning(cls_name + ' inputs must come from '
                        '`tfe.keras.Input` (thus holding past layer metadata),'
                        ' they cannot be the output of '
                        'a previous non-Input layer. '
                        'Here, a tensor specified as '
                        'input to "' + self.name + '" was not an Input tensor,'
                        ' it was generated by layer ' + layer.name + '.\n'
                        'Note that input tensors are '
                        'instantiated via `tensor = tfe.keras.Input(shape)`.\n'
                        'The tensor that caused the issue was: ' + str(x.name))

     # Check compatibility of batch sizes of Input Layers.
    input_batch_sizes = [
        training_utils.get_static_batch_size(x._keras_history.layer)
        for x in self.inputs
    ]
    consistent_batch_size = None
    for batch_size in input_batch_sizes:
      if batch_size is not None:
        if (consistent_batch_size is not None and
            batch_size != consistent_batch_size):
          raise ValueError('The specified batch sizes of the Input Layers'
                           ' are incompatible. Found batch sizes: {}'.format(
                               input_batch_sizes))
        consistent_batch_size = batch_size

    for x in self.outputs:
      if not hasattr(x, '_keras_history'):
        cls_name = self.__class__.__name__
        raise ValueError('Output tensors to a ' + cls_name + ' must be '
                         'the output of a tfe.keras `Layer` '
                         '(thus holding past layer metadata). Found: ' + str(x))
