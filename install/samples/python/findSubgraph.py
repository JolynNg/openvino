import os
import numpy as np
import openvino.runtime as ov
import csv
import re

def load_model(core, model_path, device):
    try:
        compiled_model = core.compile_model(model_path, device)
        return compiled_model
    except RuntimeError as e:
        print(f"❌ Error loading model from {model_path}: {str(e)}")
        raise

def perform_inference(compiled_model, input_data):
    try:
        infer_request = compiled_model.create_infer_request()

        # Set the input tensors
        for i, input_tensor in enumerate(input_data):
            infer_request.set_input_tensor(i, input_tensor)

        infer_request.start_async()
        infer_request.wait()

        # Handle multiple output tensors
        infer_output = compiled_model.outputs
        output_tensors = []
        for i in range(len(infer_output)):
            output_tensor = infer_request.get_output_tensor(i)
            output_tensors.append(output_tensor.data)

        return output_tensors

    except RuntimeError as e:
        print(f"❌ Error during inference: {str(e)}")
        raise

def compute_absolute_diff(cpu_results, npu_results):
    absolute_diffs = []

    # Assuming cpu_results and npu_results are lists of numpy arrays (one per output tensor)
    for cpu_tensor, npu_tensor in zip(cpu_results, npu_results):
        # Compute absolute difference
        diff = np.abs(cpu_tensor.flatten() - npu_tensor.flatten())
        absolute_diffs.extend(diff)

    return absolute_diffs

def extract_number(filename):
    match = re.search(r'_(\d+)\.xml$', filename)
    return int(match.group(1)) if match else float('inf')

def main(subgraph_folder_cpu, subgraph_folder_npu, output_csv, seed=42):
    np.random.seed(seed)  # Set the numpy random seed for reproducibility
    core = ov.Core()
    subgraph_files_cpu = [f for f in os.listdir(subgraph_folder_cpu) if f.endswith('.xml')]
    subgraph_files_npu = [f for f in os.listdir(subgraph_folder_npu) if f.endswith('.xml')]
    subgraph_files = list(set(subgraph_files_cpu) & set(subgraph_files_npu))
    
    results = []
    
    for subgraph_file in subgraph_files:
        model_path_cpu = os.path.join(subgraph_folder_cpu, subgraph_file)
        model_path_npu = os.path.join(subgraph_folder_npu, subgraph_file)
        
        # Load models for CPU and NPU
        try:
            compiled_model_cpu = load_model(core, model_path_cpu, "CPU")
            compiled_model_npu = load_model(core, model_path_npu, "NPU")
        except RuntimeError as e:
            print(f"❌ Skipping {subgraph_file} due to model loading error.")
            continue
        
        print(f"🚀 Processing {subgraph_file}...")

        # Get input information
        input_info = compiled_model_npu.inputs  # use the static dimension NPU input
        input_data = []
        
        for input in input_info:
            input_shape = input.shape
            input_type = input.element_type.to_dtype()
            input_array = np.random.rand(*input_shape).astype(input_type)
            input_tensor = ov.Tensor(array=input_array, shared_memory=True)
            input_data.append(input_tensor)
        
        # Perform CPU inference
        try:
            cpu_result = perform_inference(compiled_model_cpu, input_data)
        except RuntimeError as e:
            print(f"❌ Error during CPU inference for {subgraph_file}: {str(e)}")
            continue
        
        # Perform NPU inference
        try:
            npu_result = perform_inference(compiled_model_npu, input_data)
        except RuntimeError as e:
            print(f"❌ Error during NPU inference for {subgraph_file}: {str(e)}")
            continue
        
        # Compute absolute differences
        absolute_diffs = compute_absolute_diff(cpu_result, npu_result)
        
        # Sort the absolute differences in descending order and remove duplicates
        sorted_diffs = sorted(set(absolute_diffs), reverse=True)
        
        # Keep only the top 5 sorted absolute differences
        top_5_diffs = sorted_diffs[:5]
        
        # Round the top 5 absolute differences to 5 decimal places
        top_5_diffs_rounded = [round(diff, 5) for diff in top_5_diffs]
        
        # Format the results for CSV
        results.append([subgraph_file] + top_5_diffs_rounded)

    # Sort results by subgraph filename
    results.sort(key=lambda x: extract_number(x[0]))
    
    # Write results to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Subgraph", "Top 5 Absolute Differences"])
        writer.writerows(results)
    
    print(f"✅ Results saved to {output_csv}")

if __name__ == "__main__":
    subgraph_folder_cpu = "C:/Phi3-mini-4k-instruct-onnx/subgraph/CPU/"
    subgraph_folder_npu = "C:/Phi3-mini-4k-instruct-onnx/subgraph/NPU/"
    output_csv = "output_top_5_absolute_diffs.csv"
    
    # Set your desired numpy seed here
    numpy_seed = 0 
    
    main(subgraph_folder_cpu, subgraph_folder_npu, output_csv, seed=numpy_seed)