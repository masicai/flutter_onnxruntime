"""
This script is used to extract the SentencepieceTokenizer node from the model and save it as a new model.

ONNX model is downloaded from: https://huggingface.co/WiseIntelligence/universal-sentence-encoder-multilingual-3-onnx-quantized/blob/main/embedding_model_quantized.onnx

The extracted SentencepieceTokenizer is used in the integration test.
"""


#!/usr/bin/env python3
import onnx
from onnx import helper, numpy_helper
import numpy as np
import sys
from pathlib import Path

# Load the model
input_model_path = '/data/models/embedding_model_quantized.onnx'
output_model_path = '/tmp/sentencepiece_tokenizer_model.onnx'

print(f"Loading model from {input_model_path}")
model = onnx.load(input_model_path)

# Find the SentencepieceTokenizer node
sp_nodes = [n for n in model.graph.node if n.op_type == 'SentencepieceTokenizer']
if not sp_nodes:
    print("Error: No SentencepieceTokenizer nodes found in the model")
    sys.exit(1)

sp_node = sp_nodes[0]
original_outputs = list(sp_node.output)
print(f"Found SentencepieceTokenizer node: {sp_node.name}")
print(f"Node inputs: {list(sp_node.input)}")
print(f"Node outputs: {original_outputs}")

# Get all initializers needed for this node
initializer_names = list(sp_node.input)[1:]  # All inputs except the first one are expected to be initializers
initializers = [init for init in model.graph.initializer if init.name in initializer_names]
print(f"Found {len(initializers)} initializers needed for the SentencePiece node")

# Create a new model with just the SentencePiece node
# We will use "text" as the input name and "indices" and "output" as the output names
new_input_name = "inputs"
new_output_names = ["output", "indices"]

# Create a modified version of the node with new input/output names
modified_node = onnx.NodeProto()
modified_node.CopyFrom(sp_node)

# Modify input - just set the first input to the new name
if len(modified_node.input) > 0:
    first_input = modified_node.input[0]
    modified_node.input.remove(first_input)
    modified_node.input.insert(0, new_input_name)

# Set the outputs to the new names
while len(modified_node.output) > 0:
    modified_node.output.pop()
for new_name in new_output_names[:len(original_outputs)]:
    modified_node.output.append(new_name)

print(f"Modified node inputs: {list(modified_node.input)}")
print(f"Modified node outputs: {list(modified_node.output)}")

# Create input tensor
input_tensor = helper.make_tensor_value_info(
    new_input_name,
    onnx.TensorProto.STRING,
    [None]  # Dynamic batch dimension
)

# Create output tensors - use INT32 for the indices output
output_tensors = []
for i, name in enumerate(new_output_names[:len(original_outputs)]):
    if name == "indices":
        output_tensors.append(helper.make_tensor_value_info(
            name,
            onnx.TensorProto.INT64,  # Using INT32 instead of INT64
            [None, None]  # Dynamic dimensions
        ))
        print(f"Created {name} output with INT32 type")
    else:
        output_tensors.append(helper.make_tensor_value_info(
            name,
            onnx.TensorProto.INT32,
            [None, None]  # Dynamic dimensions
        ))
        print(f"Created {name} output with INT64 type")

# Create the new graph
new_graph = helper.make_graph(
    [modified_node],
    "sentencepiece_tokenizer",
    [input_tensor],
    output_tensors,
    initializer=initializers
)

# Create the new model
new_model = helper.make_model(
    new_graph,
    producer_name="ONNX SentencePiece Extractor"
)

# Make sure to include the necessary op domains
for opset in model.opset_import:
    new_model.opset_import.extend([opset])

# Set the IR version to match the original
new_model.ir_version = model.ir_version

# Check and save the model
try:
    onnx.checker.check_model(new_model)
    print("Model validation passed")
except Exception as e:
    print(f"Warning: Model validation failed: {e}")
    print("Attempting to save anyway...")

onnx.save(new_model, output_model_path)
print(f"Slim model saved to {output_model_path}")

# Print model size comparison
original_size = Path(input_model_path).stat().st_size / (1024 * 1024)  # MB
new_size = Path(output_model_path).stat().st_size / (1024 * 1024)  # MB
print(f"Original model size: {original_size:.2f} MB")
print(f"New model size: {new_size:.2f} MB")
print(f"Size reduction: {(original_size - new_size) / original_size * 100:.2f}%")