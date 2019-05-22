"""Includes base classes used by all layer types."""
from abc import ABC, abstractmethod

from tensorflow.python.util import nest

from tf_encrypted import get_protocol


class Layer(ABC):
  """Base layer class.

  This is the class from which all layers inherit.
  A layer is a class implementing common neural networks operations, such
  as convolution, batch norm, etc. These operations require managing weights,
  losses, updates, and inter-layer connectivity.
  Users will just instantiate a layer and then treat it as a callable.
  We recommend that descendants of `Layer` implement the following methods:
  * `__init__()`: Save configuration in member variables
  * `build()`: Called once from `__call__`, when we know the shapes of inputs
    and `dtype`.
  * `call()`: Called in `__call__` after making sure `build()` has been called
    once. Should actually perform the logic of applying the layer to the
    input tensors (which should be passed in as the first argument).
  """

  def __init__(self, trainable=True, **kwargs):

    allowed_kwargs = {
        'input_shape',
        'batch_input_shape',
        'batch_size',
        'weights',
        'activity_regularizer',
    }
    # Validate optional keyword arguments.
    for kwarg in kwargs:
      if kwarg not in allowed_kwargs:
        raise TypeError('Keyword argument not understood:', kwarg)

    self._outbound_nodes = []
    self._inbound_nodes = []

    self.trainable = trainable
    self.built = False

  @abstractmethod
  def build(self, input_shape):
    """Creates the variables of the layer (optional, for subclass implementers).
    This is a method that implementers of subclasses of `Layer`
    can override if they need a state-creation step in-between
    layer instantiation and layer call.
    This is typically used to create the weights of `Layer` subclasses.
    Arguments:
      input_shape: Instance of `TensorShape`, or list of instances of
        `TensorShape` if the layer expects a list of inputs
        (one instance per input).
    """
    self.built = True

  @abstractmethod
  def call(self, inputs):
    """This is where the layer's logic lives.
    Arguments:
        inputs: Input tensor, or list/tuple of input tensors.
    Returns:
        A tensor or list/tuple of tensors.
    """
    return inputs

  @abstractmethod
  def compute_output_shape(self, input_shape):
    """Returns the layer's output shape"""

  def __call__(self, inputs, *args, **kargs):
    """Wraps `call`, applying pre- and post-processing steps.
    Arguments:
      inputs: input tensor(s).
      *args: additional positional arguments to be passed to `self.call`.
      **kwargs: additional keyword arguments to be passed to `self.call`.
    Returns:
      Output tensor(s).
    """
    if not self.built:
      input_shapes = inputs.shape
      self.build(input_shapes)

      self.built = True

    outputs = self.call(inputs, *args, **kargs)

    return outputs

  @property
  def prot(self):
    return get_protocol()

  @property
  def outbound_nodes(self):
    return self._outbound_nodes

  @property
  def inbound_nodes(self):
    return self._inbound_nodes

  def _add_inbound_node(self,
                        input_tensors,
                        output_tensors,
                        arguments=None):
    """Internal method to create an inbound node for the layer.
    Arguments:
        input_tensors: list of input tensors.
        output_tensors: list of output tensors.
        arguments: dictionary of keyword arguments that were passed to the
            `call` method of the layer at the call that created the node.
    """
    # pylint: disable=protected-access
    inbound_layers = nest.map_structure(lambda t: t._keras_history.layer,
                                        input_tensors)
    node_indices = nest.map_structure(lambda t: t._keras_history.node_index,
                                      input_tensors)
    tensor_indices = nest.map_structure(lambda t: t._keras_history.tensor_index,
                                        input_tensors)
    # pylint: enable=protected-access

    # Create node, add it to inbound nodes.
    Node(
        self,
        inbound_layers=inbound_layers,
        node_indices=node_indices,
        tensor_indices=tensor_indices,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        arguments=arguments)

    # Update tensor history metadata.
    # The metadata attribute consists of
    # 1) a layer instance
    # 2) a node index for the layer
    # 3) a tensor index for the node.
    # The allows layer reuse (multiple nodes per layer) and multi-output
    # or multi-input layers (e.g. a layer can return multiple tensors,
    # and each can be sent to a different layer).
    for i, tensor in enumerate(nest.flatten(output_tensors)):
      tensor._keras_history = KerasHistory(self,  # pylint: disable=protected-access
                                           len(self._inbound_nodes) - 1, i)

  def _init_set_name(self, name, zero_based=True):
    if not name:
      self._name = backend.unique_object_name(
          generic_utils.to_snake_case(self.__class__.__name__),
          zero_based=zero_based)
    else:
      self._name = name

  def _name_scope(self):
    return self.name


class Node:
  """A `Node` describes the connectivity between two layers.
  Each time a layer is connected to some new input,
  a node is added to `layer._inbound_nodes`.
  Each time the output of a layer is used by another layer,
  a node is added to `layer._outbound_nodes`.
  Arguments:
      outbound_layer: the layer that takes
          `input_tensors` and turns them into `output_tensors`
          (the node gets created when the `call`
          method of the layer was called).
      inbound_layers: a list of layers, the same length as `input_tensors`,
          the layers from where `input_tensors` originate.
      node_indices: a list of integers, the same length as `inbound_layers`.
          `node_indices[i]` is the origin node of `input_tensors[i]`
          (necessary since each inbound layer might have several nodes,
          e.g. if the layer is being shared with a different data stream).
      tensor_indices: a list of integers,
          the same length as `inbound_layers`.
          `tensor_indices[i]` is the index of `input_tensors[i]` within the
          output of the inbound layer
          (necessary since each inbound layer might
          have multiple tensor outputs, with each one being
          independently manipulable).
      input_tensors: list of input tensors.
      output_tensors: list of output tensors.
      arguments: dictionary of keyword arguments that were passed to the
          `call` method of the layer at the call that created the node.
  `node_indices` and `tensor_indices` are basically fine-grained coordinates
  describing the origin of the `input_tensors`.
  A node from layer A to layer B is added to:
    - A._outbound_nodes
    - B._inbound_nodes
  """

  def __init__(self,
               outbound_layer,
               inbound_layers,
               node_indices,
               tensor_indices,
               input_tensors,
               output_tensors,
               arguments=None):
    # Layer instance (NOT a sequence)
    if isinstance(outbound_layer, (list, tuple, dict)):
      raise ValueError('`outbound_layer` should be a layer instance, '
                       'not a list, tuple, or, dict.')

    # this is the layer that takes a nested structure of input tensors
    # and turns them into a nested structure of output tensors.
    # the current node will be added to
    # the inbound_nodes of outbound_layer.
    self.outbound_layer = outbound_layer

    # The following 3 properties describe where
    # the input tensors come from: which layers,
    # and for each layer, which node and which
    # tensor output of each node.

    # Nested structure of layer instances.
    self.inbound_layers = inbound_layers
    # Nested structure of integers, 1:1 mapping with inbound_layers.
    self.node_indices = node_indices
    # Nested of integers, 1:1 mapping with inbound_layers.
    self.tensor_indices = tensor_indices

    # Following 2 properties:
    # tensor inputs and outputs of outbound_layer.

    # Nested structure of tensors. 1:1 mapping with inbound_layers.
    self.input_tensors = input_tensors
    # Nested structure of tensors, created by outbound_layer.call().
    self.output_tensors = output_tensors

    # Following 2 properties: input and output shapes.

    # Nested structure of shape tuples, shapes of input_tensors.
    self.input_shapes = nest.map_structure(lambda x: x.shape, input_tensors)
    # Nested structure of shape tuples, shapes of output_tensors.
    self.output_shapes = nest.map_structure(lambda x: x.shape, output_tensors)

    # Optional keyword arguments to layer's `call`.
    self.arguments = arguments

    # Add nodes to all layers involved.
    for layer in nest.flatten(inbound_layers):
      if layer is not None:
        # For compatibility with external Keras, we use the deprecated
        # accessor here.
        layer.outbound_nodes.append(self)
    # For compatibility with external Keras, we use the deprecated
    # accessor here.
    outbound_layer.inbound_nodes.append(self)

  def iterate_inbound(self):
    """Returns a list of tuples representing the inbound data.
    Returns:
      List of tuples like: (inbound_layer, node_index, tensor_index, tensor).
    """
    return zip(
        nest.flatten(self.inbound_layers), nest.flatten(self.node_indices),
        nest.flatten(self.tensor_indices), nest.flatten(self.input_tensors),
    )


class KerasHistory(
    collections.namedtuple('KerasHistory',
                           ['layer', 'node_index', 'tensor_index'])):
  """Tracks the Layer call that created a Tensor, for Keras Graph Networks.
  During construction of Keras Graph Networks, this metadata is added to
  each Tensor produced as the output of a Layer, starting with an
  `InputLayer`. This allows Keras to track how each Tensor was produced, and
  this information is later retraced by the `keras.engine.Network` class to
  reconstruct the Keras Graph Network.
  Attributes:
    layer: The Layer that produced the Tensor.
    node_index: The specific call to the Layer that produced this Tensor. Layers
      can be called multiple times in order to share weights. A new node is
      created every time a Tensor is called.
    tensor_index: The output index for this Tensor. Always zero if the Layer
      that produced this Tensor only has one output. Nested structures of
      Tensors are deterministically assigned an index via `nest.flatten`.
  """
  # Added to maintain memory and performance characteristics of `namedtuple`
  # while subclassing.
  __slots__ = ()
