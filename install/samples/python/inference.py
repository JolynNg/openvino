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
        print(f"Error loading model from {model_path}: {str(e)}")
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
        # print("Infer Output")
        # print(infer_output)
        output_tensors = []
        for i in range(len(infer_output)):
            output_tensor = infer_request.get_output_tensor(i)
            output_tensors.append(output_tensor.data)

        return output_tensors

    except RuntimeError as e:
        print(f"Error during inference: {str(e)}")
        raise

def compare_results(cpu_results, npu_results, accuracy, use_tol=True):
    # Assuming cpu_results and npu_results are lists of numpy arrays (one per output tensor)
    all_passed = True
    matched_count = 0
    total_elements = 0
    
    for i, (cpu, npu) in enumerate(zip(cpu_results, npu_results)):
        cpu_flat = cpu.flatten()
        npu_flat = npu.flatten()
        total_elements += len(cpu_flat)
        
        if use_tol:
            for j, (cpu_elem, npu_elem) in enumerate(zip(cpu_flat, npu_flat)):
                if np.isclose(cpu_elem, npu_elem, atol=accuracy, rtol=1e-5):
                    matched_count += 1
                else:
                    all_passed = False
                    # print(f"Element {j} in Output {i} differs: CPU={cpu_elem}, NPU={npu_elem}")
        else:
            cpu_rounded = np.round(cpu_flat, accuracy) * (10 ** accuracy)
            npu_rounded = np.round(npu_flat, accuracy) * (10 ** accuracy)
            for j, (cpu_elem, npu_elem) in enumerate(zip(cpu_rounded, npu_rounded)):
                if int(cpu_elem) == int(npu_elem):
                    matched_count += 1
                else:
                    all_passed = False
                    # print(f"Element {j} in Output {i} differs: CPU={cpu_rounded}, NPU={npu_rounded}")
        
    print("total elements : " + str(total_elements))
    print("Matched count : " + str(matched_count))
    match_percentage = int((matched_count / total_elements) * 100)
    return all_passed, match_percentage

def extract_number(filename):
    match = re.search(r'_(\d+)\.xml$', filename)
    return int(match.group(1)) if match else float('inf')

def format_tensor_elements(tensor):
    return ','.join(map(str, tensor.flatten()[:5]))

def main(subgraph_folder_cpu, subgraph_folder_npu, output_csv, tol, dp):
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
            print(f"Skipping {subgraph_file} due to model loading error.")
            continue
        
        print(f"Processing {subgraph_file}...")
        
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
            print(f"Error during CPU inference for {subgraph_file}: {str(e)}")
            continue
        
        # Perform NPU inference
        try:
            npu_result = perform_inference(compiled_model_npu, input_data)
        except RuntimeError as e:
            print(f"Error during NPU inference for {subgraph_file}: {str(e)}")
            continue
        
        # Compare results and store results
        results_label = []
        match_percentages = []
        for tol_val in tol:
            passed, match_percentage = compare_results(cpu_result, npu_result, tol_val, True)
            result_label = "passed" if passed else "failed"
            results_label.append(result_label + " " + str(match_percentage))
            
        for dp_val in dp:
            passed, match_percentage = compare_results(cpu_result, npu_result, dp_val, False)
            result_label = "passed" if passed else "failed"
            results_label.append(result_label + " " + str(match_percentage))
        
        # Track the first 5 tensor elements for each output for both CPU and NPU
        first_5_cpu = [format_tensor_elements(tensor) for tensor in cpu_result[:5]]
        first_5_npu = [format_tensor_elements(tensor) for tensor in npu_result[:5]]
        
        results.append([subgraph_file] + results_label + match_percentages + [first_5_cpu, first_5_npu])
    
    # Sort results by subgraph filename
    results.sort(key=lambda x: extract_number(x[0]))
    
    # Write results to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = tol + [str(dp_val)+"dp" for dp_val in dp]
        writer.writerow(["Subgraph"] + [str(accuracy) + " result" for accuracy in header] + ["First 5 CPU Tensors", "First 5 NPU Tensors"])
        writer.writerows(results)
    
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    subgraph_folder_cpu = "C:/Phi3-mini-4k-instruct-onnx/test/subgraph/CPU/"
    subgraph_folder_npu = "C:/Phi3-mini-4k-instruct-onnx/test/subgraph/NPU/"
    output_csv = "output.csv"
    
    # main(subgraph_folder_cpu, subgraph_folder_npu, output_csv, [1e-2, 1e-4], [2, 4])
    main(subgraph_folder_cpu, subgraph_folder_npu, output_csv, [], [4])