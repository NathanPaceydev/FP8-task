import torch
import unittest
import numpy as np
from solution import round_to_fp8_represented_as_int8, int32_to_bfloat16, undo_int8_fp8

def numpy_bfloat16_to_fp8(tensor, N):
    # Convert bfloat16 tensor to float32 for compatibility with numpy
    tensor_float32 = tensor.float()
    # Convert the float32 tensor to numpy
    tensor_np = tensor_float32.numpy().view(np.uint32)
    # Define masks for extracting parts of the float32 format
    sign_mask = 0x80000000  # 1 bit
    exponent_mask = 0x7F800000  # 8 bits of exponent in float32
    mantissa_mask = 0x007FFFFF  # 23 bits of mantissa in float32
    # Extract sign, exponent, and mantissa
    sign = (tensor_np & sign_mask) >> 31
    exponent = (tensor_np & exponent_mask) >> 23
    mantissa = tensor_np & mantissa_mask
    # Adjust the exponent from float32 to target fp8 format (assuming 5 exponent bits for this example)
    # Note: This adjustment is heuristic and may need refinement for specific cases
    exponent_offset = (1 << (8 - 1)) - (1 << (5 - 1))
    exponent = np.clip(exponent - exponent_offset, 0, (1 << 5) - 1)
    # Truncate the mantissa to N bits
    mantissa >>= (23 - N)
    # Combine sign, adjusted exponent, and truncated mantissa into fp8 format
    fp8 = (sign << (1 + 5 + N)) | (exponent << N) | mantissa
    # Since we don't have native fp8 support, for demonstration we'll store these in a uint8 numpy array
    # This means we're not returning a tensor with computational capabilities, just a representation of the fp8 values
    fp8_array = fp8.astype(np.uint8)
    return fp8_array

def numpy_fp8_to_bfloat16(fp8_tensor, N):
    # Convert the uint8 tensor to a NumPy array for processing
    fp8_np = fp8_tensor.numpy()
    # Constants for FP8 format
    fp8_e_bits = 5
    fp8_bias = (1 << (fp8_e_bits - 1)) - 1
    # Constants for bfloat16 format
    bfloat16_bias = 127
    # Extract the sign, exponent, and mantissa from FP8 format
    sign = (fp8_np >> 7) & 1
    exponent = (fp8_np >> N) & ((1 << fp8_e_bits) - 1)
    mantissa = fp8_np & ((1 << N) - 1)
    # Adjust the exponent from FP8 bias to bfloat16 bias
    exponent_bfloat16 = exponent - fp8_bias + bfloat16_bias
    # Assemble the float32 representation: we need to place the exponent and mantissa in the right position
    # and account for the 24-bit mantissa of float32
    float32_repr = (sign.astype(np.uint32) << 31) | (exponent_bfloat16.astype(np.uint32) << 23) | (mantissa.astype(np.uint32) << (23 - N))
    # Convert the uint32 representation to float32 ######### this is where the torch version breaks
    bfloat16_as_float32 = np.frombuffer(float32_repr.tobytes(), dtype=np.float32)
    # Convert the NumPy array to a PyTorch tensor and cast to bfloat16
    bfloat16_tensor = torch.from_numpy(bfloat16_as_float32).to(torch.bfloat16)
    return bfloat16_tensor


class TestFP8Conversion(unittest.TestCase):
    def setUp(self):
        self.original_tensor = torch.tensor([2.0000, 0.5000, 4.2600, 0.1250], dtype=torch.bfloat16)
        self.n_mantissa = 3  # Number of mantissa bits for FP8 format

    def test_round_to_fp8_represented_as_int8(self):
        # Convert bfloat16 tensor to FP8 format using the TorchScript function
        fp8_tensor = round_to_fp8_represented_as_int8(self.original_tensor, self.n_mantissa)
        self.assertTrue(fp8_tensor.dtype == torch.uint8, "The resulting FP8 tensor should be of type uint8.")
        # Convert bfloat16 tensor to FP8 format using the NumPy function for comparison
        fp8_tensor_np = numpy_bfloat16_to_fp8(self.original_tensor, self.n_mantissa)
        
        self.assertTrue(np.array_equal(fp8_tensor.numpy(), fp8_tensor_np), "TorchScript and NumPy implementations should produce the same FP8 tensor.")

    def test_undo_int8_fp8(self):
        # Convert the original bfloat16 tensor to FP8 format
        fp8_tensor = round_to_fp8_represented_as_int8(self.original_tensor, self.n_mantissa)
        # Convert the FP8 tensor back to bfloat16 format using the TorchScript function
        reconstructed_tensor = undo_int8_fp8(fp8_tensor, self.n_mantissa, torch.bfloat16)
        self.assertTrue(reconstructed_tensor.dtype == torch.bfloat16, "The reconstructed tensor should be of type bfloat16.")
        # Convert the FP8 tensor back to bfloat16 format using the NumPy function for comparison
        reconstructed_tensor_np = numpy_fp8_to_bfloat16(torch.tensor(fp8_tensor, dtype=torch.uint8), self.n_mantissa)
        self.assertTrue(torch.allclose(reconstructed_tensor, reconstructed_tensor_np, atol=1e-3), "TorchScript and NumPy implementations should produce similar bfloat16 tensors.")


    def test_special_values(self):
        special_values_tensor = torch.tensor([0.0, float('inf'), float('-inf'), float('nan')], dtype=torch.bfloat16)
        fp8_tensor = round_to_fp8_represented_as_int8(special_values_tensor, self.n_mantissa)
        # Expected FP8 representations for the special values
        expected_values = torch.tensor([0, 124, 252, 255], dtype=torch.uint8)
        # Check zero, positive infinity, and negative infinity directly
        self.assertTrue(torch.equal(fp8_tensor[:3], expected_values[:3]), "Zero and infinities should be correctly represented.")
        # Check NaN separately since NaN != NaN
        self.assertTrue(torch.isnan(special_values_tensor[-1]), "The original last value should be NaN.")
        self.assertEqual(fp8_tensor[-1], expected_values[-1], "NaN should be represented as 255 in FP8.")

    def test_scaling_factor_effectiveness(self):
        small_values_tensor = torch.tensor([0.01, 0.02, 0.03, 0.04], dtype=torch.bfloat16)
        max_value_fp8 = 2 ** (2 ** self.n_mantissa - 1)
        scaling_factor = -max_value_fp8 / small_values_tensor.max()
        # Convert to FP8 with scaling
        fp8_tensor = round_to_fp8_represented_as_int8(small_values_tensor, self.n_mantissa, scaling_factor=scaling_factor)
        # Undo conversion from FP8, applying the same scaling factor
        reconstructed_tensor = undo_int8_fp8(fp8_tensor, self.n_mantissa, torch.bfloat16, scaling_factor=scaling_factor)
        print("Reconstructed tensor:", reconstructed_tensor)
        # Check that the signs are correct and the values are close to the originals
        self.assertTrue(torch.all(reconstructed_tensor >= 0), "All values in the reconstructed tensor should be non-negative.")
        self.assertTrue(torch.allclose(small_values_tensor, reconstructed_tensor, atol=1e-2), "The tensor should be accurately reconstructed after applying the scaling factor.")


if __name__ == '__main__':
    unittest.main()
