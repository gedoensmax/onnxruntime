import onnx
import onnx.helper as helper
import onnx_graphsurgeon as gs
import logging
import re
import os
import argparse
import numpy as np


max_fp8 = 447



def update_model_opset(model, target_opset, domain=""):
    """
    Updates the opset version of an ONNX model using only onnx-graphsurgeon.

    Args:
        model_path (str): Path to the input ONNX model
        output_path (str): Path where the modified model will be saved
        target_opset (int): Target opset version to convert to
        domain (str, optional): ONNX domain, default is "" (main ONNX domain)

    Returns:
        bool: True if successful, False otherwise
    """


    # Update the opset import
    for opset in model.opset_import:
        if opset.domain == domain:
            opset.version = target_opset
            break
    else:
        # If we didn't find the domain, add a new opset import
        opset = model.opset_import.add()
        opset.domain = domain
        opset.version = target_opset

    return model

def convert_opset_to_21_proto(model_proto: onnx.ModelProto):
    """Modify the model's opset to 21 if it's not already, operating on a ModelProto.

    Args:
        model_proto (ModelProto): The ONNX model proto to modify.

    Returns:
        ModelProto: The updated ONNX model proto with opset version 21.

    """
    current_opset = {opset.domain: opset.version for opset in model_proto.opset_import}

    default_domain_version = current_opset.get("", 0)
    if default_domain_version >= 21:
        logging.info(
            "Model already uses opset version %s for the default domain. Skip conversion.",
            default_domain_version,
        )
        return model_proto  # No conversion needed

    new_opset_imports = [
        helper.make_opsetid("", 21),  # Default domain with opset version 21
        helper.make_opsetid("com.microsoft", 1),  # Microsoft domain with version 1
    ]

    for domain, version in current_opset.items():
        if domain not in ["", "com.microsoft"]:
            new_opset_imports.append(helper.make_opsetid(domain, version))

    # for node in model_proto.graph.node:
    #     if node.op_type == "Conv":
    #         node.domain =

    # Update the model's opset imports
    model_proto.ClearField("opset_import")
    model_proto.opset_import.extend(new_opset_imports)

    logging.info("Model opset successfully converted to 21.")

    return model_proto

def trt_quantize_to_onnx_quantize(node, dtype, use_onnx_standard_quant):
    if use_onnx_standard_quant:
        node.domain = None
        node.op = node.op[len("TRT_FP8"):]

        fp8_zero = gs.Constant(
            name=f"{node.name}/fp8_zero",
            values=np.zeros(shape=(1,), dtype=np.float32),
            export_dtype=onnx.TensorProto.FLOAT8E4M3FN,
        )
        # quant_scale = gs.Constant(
        #     name=f"{node.name}/quant_scale",
        #     values=np.ones(shape=(1,), dtype=np.float32),
        #     export_dtype=dtype,
        # )
        quant_scale = node.inputs[1]
        # print(quant_scale)
        # quant_scale.values = quant_scale.values.astype(dtype)
        node.inputs = [node.inputs[0], quant_scale, fp8_zero]
    else:
        quant_scale = node.inputs[1]
        quant_scale.values = quant_scale.values.astype(dtype)
        node.inputs = [node.inputs[0], quant_scale]


def main(onnx_file, out_file, use_onnx_standard_quant):

    model_onnx = onnx.load(onnx_file)
    # first infer shapes and then simplify to remove shape ops
    graph = gs.import_onnx(model_onnx)
    softmax_nodes = []
    nodes_to_remove = []
    nodes_tocast = []
    network_type = graph.inputs[0].dtype
    if network_type == 'float16':
        network_type = onnx.TensorProto.FLOAT16
    elif network_type == 'float':
        network_type = onnx.TensorProto.FLOAT
    else:
        network_type = onnx.TensorProto.BFLOAT16
        # network_type = onnx.TensorProto.FLOAT



    print(f"Using type {graph.inputs[0].dtype}:{network_type}")
    print(f"Using standard ops {use_onnx_standard_quant}")

    for node in graph.nodes:
        if node.op.startswith("TRT_"):
            trt_quantize_to_onnx_quantize(node, network_type, use_onnx_standard_quant)


    graph.toposort()
    # if use_onnx_standard_quant:
    #     graph.fold_constants()
    graph.cleanup()

    model_onnx = gs.export_onnx(graph)
    # model_onnx = convert_opset_to_21_proto(model_onnx)
    model_onnx = update_model_opset(model_onnx, 21)
    onnx.save(
        model_onnx,
        out_file,
        save_as_external_data=True,
        location=os.path.basename(out_file) + "_data",
    )
    print(f"ONNX model '{out_file}' saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ONNX surgeon for full export')
    parser.add_argument('onnx_file', type=str, help='Input ONNX model')
    parser.add_argument('-o', type=str, help="Output ONNX model")

    args = parser.parse_args()


    main(args.onnx_file, args.o, use_onnx_standard_quant=True)

