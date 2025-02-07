import openvino.runtime as ov
import numpy as np
import csv
import xml.etree.ElementTree as ET
import pandas as pd
import os

def ov_type_to_numpy_type(ov_type):
    # Map OpenVINO types to NumPy types
    type_mapping = {
        ov.Type.f32: np.float32,
        ov.Type.f64: np.float64,
        ov.Type.i32: np.int32,
        ov.Type.i64: np.int64,
        ov.Type.u8: np.uint8,
        ov.Type.u16: np.uint16,
    }
    return type_mapping.get(ov_type, None)

def load_model(device_name, model_xml, model_bin, target_shape):
    # Load the model
    core = ov.Core()
    model = core.read_model(model=model_xml, weights=model_bin)
    
    # Reshape the model if necessary
    for input_tensor in model.inputs:
        if input_tensor.partial_shape.is_dynamic:
            print(f"Input {input_tensor.any_name} has a dynamic shape: {input_tensor.partial_shape}")
            model.reshape({input_tensor.any_name: target_shape})
            print(f"Model reshaped to accept input shape: {target_shape}")

    # Compile the model
    compiled_model = core.compile_model(model, device_name)
    return compiled_model

def perform_inference(compiled_model, input_data):
    # Create an inference request
    infer_request = compiled_model.create_infer_request()

    # Check input compatibility
    for input_index, input_tensor in enumerate(compiled_model.inputs):
        target_shape = input_tensor.shape
        if input_data.shape != target_shape:
            print(f"Reshaping input data from {input_data.shape} to {target_shape}")
            input_data = np.reshape(input_data, target_shape)

        # Convert input data type if necessary
        input_dtype = np.float32  # Assuming the model uses float32
        if input_data.dtype != input_dtype:
            print(f"Converting input data from {input_data.dtype} to {input_dtype}")
            input_data = input_data.astype(input_dtype)

        # Set input tensor
        ov_input_tensor = ov.Tensor(input_data)
        infer_request.set_input_tensor(input_index, ov_input_tensor)

    # Perform inference
    infer_request.start_async()
    infer_request.wait()

    # Get all output tensors
    output_buffers = [infer_request.get_output_tensor(i).data for i in range(len(compiled_model.outputs))]

    return output_buffers

def compare_tensors(cpu_tensor, npu_tensor, writer):
    # Check if the tensors have the same shape
    if cpu_tensor.shape != npu_tensor.shape:
        writer.writerow(['Shape Mismatch', cpu_tensor.shape, npu_tensor.shape, '', '', ''])
        return

    # Flatten the tensors to make element-wise comparison easier
    cpu_flattened = cpu_tensor.flatten()
    npu_flattened = npu_tensor.flatten()

    # Compare each element, rounding to 4 decimal places, and save results in the specified format
    for idx, (cpu_val, npu_val) in enumerate(zip(cpu_flattened, npu_flattened)):
        # Round the values to 4 decimal places for comparison
        cpu_val_rounded = round(cpu_val, 4)
        npu_val_rounded = round(npu_val, 4)

        # Calculate absolute difference and round it to 4 decimal places
        abs_diff = round(abs(cpu_val_rounded - npu_val_rounded), 4)

        # Save result only if AbsDiff is not 0
        if abs_diff > 0:
            # Write results with explicit formatting for 4 decimal places
            writer.writerow([f"Element {idx + 1}: CPU={cpu_val_rounded:.4f}, NPU={npu_val_rounded:.4f}, AbsDiff={abs_diff:.4f}"])

def main():
    # Load the CPU model
    model_cpu_xml = "C:/Phi3-mini-4k-instruct-onnx/subgraph/CPU/OpenVINO-EP-subgraph_3.xml"
    model_cpu_bin = "C:/Phi3-mini-4k-instruct-onnx/subgraph/CPU/OpenVINO-EP-subgraph_3.bin"

    # Load the NPU model
    model_npu_xml = "C:/Phi3-mini-4k-instruct-onnx/test/subgraph/NPU/OpenVINO-EP-subgraph_3-cut-Sqrt_4338.xml"
    model_npu_bin = "C:/Phi3-mini-4k-instruct-onnx/test/subgraph/NPU/OpenVINO-EP-subgraph_3-cut-Sqrt_4338.bin"
    
    # Extract the filename without extension
    base_name = os.path.splitext(os.path.basename(model_npu_xml))[0]
    
    # Create CSV file name based on the XML filename
    csv_filename = f"{base_name}_failed_elements.csv"

    np.random.seed(911)

    # Dummy input data
    input_shape = (1, 1, 3072)  # Adjust based on your model
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    print("Initial dummy input data type:", dummy_input.dtype)

    # Load and compile the model for CPU
    compiled_model_cpu = load_model("CPU", model_cpu_xml, model_cpu_bin, input_shape)
    print("\nModel compiled on device: CPU")

    # Perform inference
    output_cpu = perform_inference(compiled_model_cpu, dummy_input)
    print("Inference results on CPU:", output_cpu)

    # Load and compile the model for NPU
    compiled_model_npu = load_model("NPU", model_npu_xml, model_npu_bin, input_shape)
    print("\nModel compiled on device: NPU")

    # Perform inference
    output_npu = perform_inference(compiled_model_npu, dummy_input)
    print("Inference results on NPU:", output_npu)
    
    # Open the CSV file to save the comparison results
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Compare the results
        print("\nComparing CPU and NPU outputs...")
        print(f"Length of output_cpu: {len(output_cpu)}")
        print(f"Length of output_npu: {len(output_npu)}")
        for tensor_idx, (cpu_tensor, npu_tensor) in enumerate(zip(output_cpu, output_npu)):
            print(f"\nComparing Output tensor {tensor_idx}...")
            compare_tensors(cpu_tensor, npu_tensor, writer)

    print(f"\nLayer names saved to '{csv_filename}'.")

if __name__ == "__main__":
    main()