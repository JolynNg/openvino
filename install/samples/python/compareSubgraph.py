import openvino.runtime as ov
import numpy as np
import csv
import os

def inspect_model(model_xml):
    """Inspect and print model output layer names and shapes."""
    core = ov.Core()
    model = core.read_model(model=model_xml)
    print(f"\n🔍 Inspecting model: {model_xml}")

    for output_tensor in model.outputs:
        output_name = output_tensor.any_name
        output_shape = output_tensor.partial_shape
        
        if output_shape.is_dynamic:
            print(f"➞ Output tensor: {output_name}, Shape: ⚠️ Dynamic Shape ({output_shape})")
        else:
            print(f"➞ Output tensor: {output_name}, Shape: {output_shape.to_shape()}")

def load_model(device_name, model_xml, model_bin, target_shape):
    """Load and compile an OpenVINO model for a given device."""
    core = ov.Core()
    model = core.read_model(model=model_xml, weights=model_bin)
    
    print(f"\n🚀 Loading model on {device_name}: {model_xml}")

    for input_tensor in model.inputs:
        input_name = input_tensor.any_name
        input_partial_shape = input_tensor.partial_shape

        if input_partial_shape.is_dynamic:
            print(f"⚠️ [Reshaping] {input_name} has a dynamic shape: {input_partial_shape}")
            model.reshape({input_name: target_shape})
            print(f"➞ Model reshaped to: {target_shape}")
        else:
            print(f"➞ Input {input_name} has fixed shape: {input_partial_shape.to_shape()}")

    return core.compile_model(model, device_name)

def perform_inference(compiled_model, input_data):
    """Perform inference and return all output tensors."""
    infer_request = compiled_model.create_infer_request()

    for input_index, input_tensor in enumerate(compiled_model.inputs):
        target_shape = input_tensor.shape
        if input_data.shape != target_shape:
            print(f"🔄 Reshaping input from {input_data.shape} to {target_shape}")
            input_data = np.reshape(input_data, target_shape)

        if input_data.dtype != np.float32:
            print(f"🔄 Converting input data from {input_data.dtype} to np.float32")
            input_data = input_data.astype(np.float32)

        ov_input_tensor = ov.Tensor(input_data)
        infer_request.set_input_tensor(input_index, ov_input_tensor)

    infer_request.start_async()
    infer_request.wait()

    # Explicitly return all outputs as a list
    return [(i, infer_request.get_output_tensor(i).data) for i in range(len(compiled_model.outputs))]

def compare_tensors(cpu_tensor, npu_tensor, writer, model_name):
    """Compare CPU and NPU inference results, extract highest absolute differences, and save to CSV in a clean format."""
    cpu_shape, npu_shape = cpu_tensor.shape, npu_tensor.shape

    print(f"\n🔍 Comparing outputs for {model_name}: CPU {cpu_shape}, NPU {npu_shape}")

    if cpu_shape != npu_shape:
        writer.writerow([model_name, "[Shape mismatch]", "[N/A]", "[N/A]"])
        return

    abs_diff = np.abs(cpu_tensor.flatten() - npu_tensor.flatten())

    if len(abs_diff) == 0:
        writer.writerow([model_name, "[No data]", "[N/A]", "[N/A]"])
        return

    sorted_indices = np.argsort(-abs_diff)[:5]  # Get indices of top 5 differences
    top_diffs = abs_diff[sorted_indices]
    top_cpu_values = cpu_tensor.flatten()[sorted_indices]  # Get corresponding CPU values
    top_npu_values = npu_tensor.flatten()[sorted_indices]  # Get corresponding NPU values

    # Convert NumPy types to native Python floats and format as lists
    top_diffs_str = f"[{', '.join(map(lambda x: f'{x:.5f}', top_diffs))}]"
    top_cpu_values_str = f"[{', '.join(map(lambda x: f'{x:.5f}', top_cpu_values))}]"
    top_npu_values_str = f"[{', '.join(map(lambda x: f'{x:.5f}', top_npu_values))}]"

    print(f"\n📌 {model_name}")
    print(f"   🔴 Top 5 Absolute Differences: {top_diffs_str}")
    print(f"   🔵 Corresponding CPU Values: {top_cpu_values_str}")
    print(f"   🟢 Corresponding NPU Values: {top_npu_values_str}")

    writer.writerow([model_name, top_diffs_str, top_cpu_values_str, top_npu_values_str])

def main():
    npu_folder = "C:/Phi3-mini-4k-instruct-onnx/subgraph/test"

    np.random.seed(0)
    input_shape = (1, 1, 3072)
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    csv_filename = "npu_vs_cpu_absolute_diff.csv"

    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Subgraph", "Top 5 Absolute Differences", "CPU Values", "NPU Values"])

        for filename in os.listdir(npu_folder):
            if filename.endswith(".xml"):
                model_npu_xml = os.path.join(npu_folder, filename)
                model_npu_bin = model_npu_xml.replace(".xml", ".bin")

                if not os.path.exists(model_npu_bin):
                    print(f"⏭️ Skipping {model_npu_xml}: BIN file not found.")
                    continue

                print(f"\n🚀 Processing NPU subgraph: {filename}")

                inspect_model(model_npu_xml)

                compiled_model_cpu = load_model("CPU", model_npu_xml, model_npu_bin, input_shape)
                compiled_model_npu = load_model("NPU", model_npu_xml, model_npu_bin, input_shape)

                output_cpu = perform_inference(compiled_model_cpu, dummy_input)
                output_npu = perform_inference(compiled_model_npu, dummy_input)

                for (_, cpu_tensor), (_, npu_tensor) in zip(output_cpu, output_npu):  # Extract only tensors
                    compare_tensors(cpu_tensor, npu_tensor, writer, filename)

    print(f"\n✅ All comparisons saved to {csv_filename}")

if __name__ == "__main__":
    main()
