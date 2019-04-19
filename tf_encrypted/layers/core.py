from typing import Optional, List, Any, Union
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
from ..protocol.protocol import get_protocol, Protocol
from ..protocol.pond import PondPublicTensor, PondPrivateTensor, TFEVariable

# TODO
# Split backward function in compute_gradient and compute_backpropagated_error?


class Layer(ABC):

    def __init__(self, input_shape: List[int]) -> None:
        self.input_shape = input_shape
        self.output_shape = self.get_output_shape()
        self.layer_output = None

    @abstractmethod
    def get_output_shape(self) -> List[int]:
        """Returns the layer's output shape"""

    @abstractmethod
    def initialize(
        self,
        *args: Optional[Union[np.ndarray, tf.Tensor, PondPublicTensor, PondPrivateTensor]],
        **kwargs: Optional[Union[np.ndarray, tf.Tensor, PondPublicTensor, PondPrivateTensor]]
    ) -> None:
        pass

    @abstractmethod
    def call(
        self,
        *args: Any,
        **kwargs: Any
    ) -> Optional[TFEVariable]:
        """
        Performs a single forward pass for the layer, building it if needed.
        """
        pass

    @abstractmethod
    def backward(
        self,
        *args: Any,
        **kwargs: Any
    ) -> Optional[TFEVariable]:
        """
        Performs the derivative for this layer, according to the chain rule.
        """
        pass

    @property
    def prot(self) -> Optional[Protocol]:
        return get_protocol()
