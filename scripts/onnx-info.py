#!/usr/bin/env python3

import sys
import onnx
from onnx import numpy_helper


def get_tensor_shape(tensor_type):
    shape = []
    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            shape.append(dim.dim_param)
        elif dim.dim_value:
            shape.append(dim.dim_value)
        else:
            shape.append("?")
    return shape


def get_dtype(elem_type):
    return onnx.TensorProto.DataType.Name(elem_type)


def print_model_info(model):
    print("=" * 50)
    print("MODEL INFO")
    print("=" * 50)

    print(f"IR version: {model.ir_version}")
    print(f"Producer: {model.producer_name}")
    print(f"Producer version: {model.producer_version}")

    if model.opset_import:
        for opset in model.opset_import:
            print(f"Opset domain: {opset.domain or 'ai.onnx'} version: {opset.version}")

    print("\n" + "=" * 50)
    print("INPUTS")
    print("=" * 50)

    for inp in model.graph.input:
        t = inp.type.tensor_type
        shape = get_tensor_shape(t)
        dtype = get_dtype(t.elem_type)
        print(f"- {inp.name}")
        print(f"  dtype: {dtype}")
        print(f"  shape: {shape}")

    print("\n" + "=" * 50)
    print("OUTPUTS")
    print("=" * 50)

    for out in model.graph.output:
        t = out.type.tensor_type
        shape = get_tensor_shape(t)
        dtype = get_dtype(t.elem_type)
        print(f"- {out.name}")
        print(f"  dtype: {dtype}")
        print(f"  shape: {shape}")

    print("\n" + "=" * 50)
    print("PARAMETERS")
    print("=" * 50)

    total_params = 0
    for initializer in model.graph.initializer:
        arr = numpy_helper.to_array(initializer)
        total_params += arr.size

    print(f"Total parameters: {total_params:,}")

    print("\n" + "=" * 50)
    print("NODES")
    print("=" * 50)

    print(f"Total nodes: {len(model.graph.node)}")
    op_types = {}
    for node in model.graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1

    for op, count in sorted(op_types.items()):
        print(f"{op}: {count}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python onnx-info.py <model.onnx>")
        sys.exit(1)

    model_path = sys.argv[1]

    try:
        model = onnx.load(model_path)
        print_model_info(model)
    except Exception as e:
        print(f"Error loading model: {e}")


if __name__ == "__main__":
    main()