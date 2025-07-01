
import numpy as np
import tensorflow as tf

class PositionalEncodingBlock:
    """
    Building the Positional Encoding Block.

    # Formula to Calculate the Positional Encoding of the any token. 
    - PE(pos, 2i) = sin(pos/10000^2i/d_model)
        where, i represent the which dimension.
        d -> model dimension. 
        This sin function is used to calculate the position of the even dimension. 
    - PE(pos, 2i+1) = cos(pos/10000^2i/d_model)
        This cos function is used to caculate the position of the odd dimension.
    In the case, positional encodcing is directly added to the input dimension instead of concatenation because concatentation increase the dimension.

    context_length: 20
    dimension_depth: 50 
    """
    def __init__(self, context_length, depth_dim):
        self.context_length = context_length
        self.depth_dim = depth_dim
    
    def positional_encoding(self):
        # dimension is taken half because half of the even dimension or depth is calculated by the sin and half of the depth is calculated by the cos.
        depth = self.depth_dim/2
        
        positions = np.arange(self.context_length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :]/depth
        
        angle_rates = 1 / (10000 **depths) 
        angle_rads = positions * angle_rates 

        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis = -1
        )

        return tf.cast(pos_encoding, dtype=tf.float32)
