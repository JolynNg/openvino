import openvino.runtime as ov
import os

def get_model_nodes(model_xml):
    """Extracts all node names from an OpenVINO model."""
    core = ov.Core()
    model = core.read_model(model=model_xml)

    node_names = set()
    for op in model.get_ops():
        node_names.add(op.get_friendly_name())  # Get operation name

    return node_names

def main():
    # Paths for CPU model
    model_cpu_xml = "C:/Phi3-mini-4k-instruct-onnx/subgraph/CPU/OpenVINO-EP-subgraph_3.xml"

    # Folder containing NPU models
    npu_folder = "C:/Phi3-mini-4k-instruct-onnx/test/subgraph/NPU"

    # Get all CPU node names
    cpu_nodes = get_model_nodes(model_cpu_xml)
    print(f"\n🔍 CPU Model Nodes ({len(cpu_nodes)} total):")
    print(cpu_nodes)

    # Iterate over all XML files in the NPU folder
    for filename in os.listdir(npu_folder):
        if filename.endswith(".xml"):
            model_npu_xml = os.path.join(npu_folder, filename)
            
            # Get all NPU node names
            npu_nodes = get_model_nodes(model_npu_xml)
            print(f"\n🔍 NPU Model Nodes ({filename}) ({len(npu_nodes)} total):")
            print(npu_nodes)

            # Find common nodes (intersection)
            common_nodes = cpu_nodes.intersection(npu_nodes)
            print(f"\n🔗 Common Nodes ({len(common_nodes)} total):")
            print(common_nodes)

if __name__ == "__main__":
    main()