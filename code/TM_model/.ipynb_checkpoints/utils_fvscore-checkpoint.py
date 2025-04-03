import os
import torch
import torch.nn as nn

from typing import Callable, Tuple, Union, Tuple, Union, Any


def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    print(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)



def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def selective_scan_flop_jit(inputs, outputs, backend="prefixsum", verbose=True):
    if verbose:
        print_jit_input_names(inputs)
    flops_fn = flops_selective_scan_ref if backend == "naive" else flops_selective_scan_fn
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops


# used for print flops
class FLOPs:
    @staticmethod
    def register_supported_ops():
        #build = import_abspy("models", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/"))
        # selective_scan_flop_jit: Callable = selective_scan_flop_jit
        # flops_selective_scan_fn: Callable = build.vmamba.flops_selective_scan_fn
        # flops_selective_scan_ref: Callable = build.vmamba.flops_selective_scan_ref
        from fvcore.nn.jit_analysis import Counter
        from fvcore.nn.jit_handles import get_shape, conv_flop_count
        def causal_conv_1d_jit(inputs, outputs):
            """
            https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
            x: (batch, dim, seqlen) weight: (dim, width) bias: (dim,) out: (batch, dim, seqlen)
            out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
            """
            # from fvcore.nn.jit_handles import conv_flop_jit
            # return conv_flop_jit(inputs, outputs)
            # for t in inputs:
                # print(t.shape if torch.is_tensor(t) else t)
            x, w = inputs[:2]
            x_shape, w_shape, out_shape = (get_shape(x), get_shape(w), get_shape(outputs[0]))
            print(x_shape, w_shape, out_shape)
            transposed = False

            # use a custom name instead of "_convolution"
            return Counter(
                {"conv": conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)}
            )
        
        supported_ops = {
            "aten::gelu": None,  # as relu is in _IGNORED_OPS
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
            "prim::PythonOp.CausalConv1dFn": causal_conv_1d_jit,
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,  # latter
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,  # latter
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,  # latter
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,  # latter
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,  # latter
            "prim::PythonOp.SelectiveScanCuda": selective_scan_flop_jit,  # latter
            # "aten::scaled_dot_product_attention": ...
        }
        return supported_ops

    @staticmethod
    def check_operations(model: nn.Module, inputs=None, input_shape=(3, 224, 224)):
        from fvcore.nn.jit_analysis import _get_scoped_trace_graph, _named_modules_with_dup, Counter, JitModelAnalysis

        if inputs is None:
            assert input_shape is not None
            if len(input_shape) == 1:
                input_shape = (1, 3, input_shape[0], input_shape[0])
            elif len(input_shape) == 2:
                input_shape = (1, 3, *input_shape)
            elif len(input_shape) == 3:
                input_shape = (1, *input_shape)
            elif len(input_shape) == 4:
                input_shape = (1, *input_shape)
            else:
                assert len(input_shape) == 5

            inputs = (torch.randn(input_shape).to(next(model.parameters()).device),)

        model.eval()

        flop_counter = JitModelAnalysis(model, inputs)
        flop_counter._ignored_ops = set()
        flop_counter._op_handles = dict()
        assert flop_counter.total() == 0  # make sure no operations supported
        print(flop_counter.unsupported_ops(), flush=True)
        print(f"supported ops {flop_counter._op_handles}; ignore ops {flop_counter._ignored_ops};", flush=True)

    @classmethod
    def fvcore_flop_count(cls, model: nn.Module, inputs=None, input_shape=(3, 224, 224), show_table=False,
                          show_arch=False, verbose=True):
        supported_ops = cls.register_supported_ops()
        from fvcore.nn.parameter_count import parameter_count as fvcore_parameter_count
        from fvcore.nn.flop_count import flop_count, FlopCountAnalysis, _DEFAULT_SUPPORTED_OPS
        from fvcore.nn.print_model_statistics import flop_count_str, flop_count_table
        from fvcore.nn.jit_analysis import _IGNORED_OPS
        from fvcore.nn.jit_handles import get_shape, addmm_flop_jit

        if inputs is None:
            assert input_shape is not None
            if len(input_shape) == 1:
                input_shape = (1, 3, input_shape[0], input_shape[0])
            elif len(input_shape) == 2:
                input_shape = (1, 3, *input_shape)
            elif len(input_shape) == 3:
                input_shape = (1, *input_shape)
            elif len(input_shape) == 4:
                input_shape = (1, *input_shape)
            else:
                assert len(input_shape) == 5

            inputs = (torch.randn(input_shape).to(next(model.parameters()).device),)

        model.eval()

        Gflops, unsupported = flop_count(model=model, inputs=inputs, supported_ops=supported_ops)

        flops_table = flop_count_table(
            flops=FlopCountAnalysis(model, inputs).set_op_handle(**supported_ops),
            max_depth=100,
            activations=None,
            show_param_shapes=True,
        )

        flops_str = flop_count_str(
            flops=FlopCountAnalysis(model, inputs).set_op_handle(**supported_ops),
            activations=None,
        )

        if show_arch:
            print(flops_str)

        if show_table:
            print(flops_table)

        params = fvcore_parameter_count(model)[""]
        flops = sum(Gflops.values())

        if verbose:
            print(Gflops.items())
            print("GFlops: ", flops, "Params: ", params, flush=True)

        return params, flops
