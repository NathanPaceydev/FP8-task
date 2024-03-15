from typing import Optional
import torch


@torch.jit.script
def round_to_fp8_represented_as_int8(
    t: torch.Tensor, 
    n_mantissa: int,
    out: Optional[torch.Tensor] = None,
    scaling_factor: float = 1,
    ) -> torch.Tensor:
    """
    Converts a tensor to an 8-bit floating-point representation (FP8) by applying scaling and quantization.
    
    Parameters:
    - t: The input tensor to be converted.
    - n_mantissa: The number of bits allocated for the mantissa in the FP8 representation.
    - out: Optional. An existing tensor to store the result. Currently unused.
    - scaling_factor: A factor to scale the tensor values before conversion. Useful for adjusting the dynamic range.
    
    Returns:
    - An 8-bit unsigned integer tensor representing the input tensor in FP8 format.
    """
    # Apply the scaling factor to the tensor
    t = t * scaling_factor
    # Define the smallest positive normal number for float32 to identify subnormals
    float32_min_normal = 1.17549435e-38

    # Check for NaN, infinity, and subnormal values
    is_nan = torch.isnan(t)
    is_inf = torch.isinf(t)
    is_subnormal = torch.abs(t) < float32_min_normal
    
    # Initialize the FP8 tensor
    fp8_value_uint8 = torch.zeros_like(t, dtype=torch.uint8)

    # Process non-special (normal) values
    normal_mask = ~(is_nan | is_inf | is_subnormal)
    if torch.any(normal_mask):
        # Extract the normal values and their absolute values
        t_normal = t[normal_mask]
        abs_tensor = torch.abs(t_normal.float())
        # Compute the exponent for the normal values
        exponent_approx = torch.floor(torch.log2(abs_tensor)) + 127
        exponent_fp8 = torch.clamp(exponent_approx - (127 - 15), min=0, max=(1 << 5) - 1)
        # Scale the absolute tensor values to simulate shifting the mantissa bits for the FP8 format
        scaled_abs_tensor = abs_tensor * (2 ** (23 - n_mantissa))

        # Separate the scaled values into their integer and fractional parts
        integer_part = torch.floor(scaled_abs_tensor)
        fractional_part = scaled_abs_tensor - integer_part

        # Determine whether to round up based on the fractional part using stochastic rounding
        should_round_up = torch.bernoulli(fractional_part).to(torch.int32)

        # Calculate the mantissa for FP8 format and apply rounding
        mantissa_approx = (integer_part % (2 ** n_mantissa)) + should_round_up
        mantissa_approx = torch.clamp(mantissa_approx, max=(1 << n_mantissa) - 1)

        # Extract the sign from the original tensor values
        sign = t_normal.sign()
        sign[sign > 0] = 0  # Positive values get a sign bit of 0
        sign[sign < 0] = 1  # Negative values get a sign bit of 1

        # Construct the FP8 value by combining sign, exponent, and mantissa
        fp8_value = (sign.to(torch.int32) << 7) | (exponent_fp8.to(torch.int32) << (n_mantissa)) | mantissa_approx.to(torch.int32)

        # Assign the FP8 value to the corresponding positions in the output tensor
        fp8_value_uint8[normal_mask] = fp8_value.to(torch.uint8)
        
    # Handle special cases
    # Represent NaNs with exponent and mantissa bits all set to 1
    if torch.any(is_nan):
        fp8_value_uint8[is_nan] = 0xFF
    # Represent infinities with exponent bits all set to 1 and mantissa bits all set to 0
    # Preserve the sign bit for positive and negative infinities
    if torch.any(is_inf):
        inf_sign = t[is_inf].sign()
        inf_sign[inf_sign > 0] = 0
        inf_sign[inf_sign < 0] = 1
        fp8_value_uint8[is_inf] = (inf_sign.to(torch.uint8) << 7) | 0x7C

    return fp8_value_uint8


@torch.jit.script
def int32_to_bfloat16(int32_tensor: torch.Tensor) -> torch.Tensor:
    # Ensure tensor is int32 for bitwise operations
    int32_tensor = int32_tensor.to(torch.int32)
    # Initialize the output tensor
    bfloat16_tensor = torch.empty(int32_tensor.size(0), dtype=torch.bfloat16)
    for i in range(int32_tensor.size(0)):
        value = int32_tensor[i]
        # Extract the sign (1 bit), exponent (8 bits), and mantissa (7 bits) from the int32 value
        sign = (value >> 31) & 1
        exponent = (value >> 23) & 0xFF
        mantissa = (value >> 16) & 0x7F
        # Construct the bfloat16 value using floating-point arithmetic
        # Note: This construction is an approximation and may not be precise
        bfloat16_value = (-1.0) ** sign * 2.0 ** (exponent - 127) * (1.0 + mantissa / 128.0)
        # Assign the constructed value to the output tensor
        bfloat16_tensor[i] = bfloat16_value
    return bfloat16_tensor


# assuming that special cases only be handeled by
@torch.jit.script
def undo_int8_fp8(
    fp8_tensor: torch.Tensor, 
    n_mantissa: int,
    target_dt: torch.dtype,
    out: Optional[torch.Tensor] = None,
    scaling_factor: float = 1,
    ) -> torch.Tensor:
    """
    Reconstructs a tensor from its 8-bit floating-point (FP8) representation.
    
    Parameters:
    - fp8_tensor: The FP8 representation of the tensor as an 8-bit unsigned integer tensor.
    - n_mantissa: The number of bits allocated for the mantissa in the FP8 representation.
    - target_dt: The data type of the output tensor (e.g., torch.bfloat16).
    - out: Optional. An existing tensor to store the result. Currently unused.
    - scaling_factor: The scaling factor to be applied to the reconstructed values.
    
    Returns:
    - A tensor of the specified data type reconstructed from the FP8 representation.
    """
    # Constants for FP8 format
    fp8_e_bits: int = 5
    fp8_bias: int = (1 << (fp8_e_bits - 1)) - 1
    # Constants for bfloat16 format
    bfloat16_bias: int = 127
    # Ensure fp8_tensor is treated as uint8 for bitwise operations
    fp8_uint = fp8_tensor.to(torch.uint8)
    # Extract sign, exponent, and mantissa from FP8 format
    sign = (fp8_uint >> 7) & 1
    exponent = (fp8_uint >> n_mantissa) & ((1 << fp8_e_bits) - 1)
    mantissa = fp8_uint & ((1 << n_mantissa) - 1)
    # Adjust the exponent from FP8 bias to bfloat16 bias
    exponent_adjusted = exponent.to(torch.int32) - fp8_bias + bfloat16_bias
    # Ensure exponent is not negative after adjustment (clamp to 0)
    exponent_adjusted = torch.clamp(exponent_adjusted, min=0)
    # Construct the float32 values from adjusted exponent (mantissa is assumed to be 0)
    # Shift the adjusted exponent into the correct position for float32 representation
    float32_values = (sign.to(torch.int32) << 31) | (exponent_adjusted << 23)
    # Convert to float32 to apply exponent and sign
    int32_tensor = float32_values.to(torch.bfloat16)
    # view_as_float not supported anymore
    bfloat16_tensor = int32_to_bfloat16(int32_tensor)
    # Cast to bfloat16
    # Apply the scaling factor to reverse the scaling applied during FP8 conversion
    bfloat16_tensor /= scaling_factor
    
    return bfloat16_tensor