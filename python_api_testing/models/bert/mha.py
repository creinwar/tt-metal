import math
from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")


import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import numpy as np

import ll_buda_bindings.ll_buda_bindings._C as _C
from utility_functions import pad_activation, pad_weight, tilize_to_list, untilize, nearest_32, print_diff_argmax, tt2torch, get_FR, set_FR, tt2torch_rm
from fused_ops.linear import Linear as TtLinear
from fused_ops.softmax import softmax

from transformers.models.bert.modeling_bert import debug_state as DS

def mha(qw, qb, kw, kb, vw, vb, hidden_dim, num_heads, device):
    assert isinstance(num_heads, int) and num_heads > 0

    QProjection = TtLinear(
        hidden_dim, hidden_dim, qw, qb, device
    )
    KProjection = TtLinear(
        hidden_dim, hidden_dim, kw, kb, device
    )
    VProjection = TtLinear(
        hidden_dim, hidden_dim, vw, vb, device
    )

    # Used to scale down the input to the softmax
    reciprocal_of_sqrt_hidden_dim_tensor = _C.tensor.Tensor(
        [1 / math.sqrt(hidden_dim)] + [0 for _ in range(32 * 32 - 1)],
        [1, 1, 32, 32],
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device
    )

    def make_attention_heads(x):
        if num_heads == 1:
            return x
        else:
            # ref code from modeling_bert.py:
            #    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
            #        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            #        x = x.view(new_x_shape)
            #        return x.permute(0, 2, 1, 3)
            untilized_x = _C.tensor.untilize(x)
            if 1:
                unt_torch = tt2torch_rm(untilized_x)
                new_shape = (x.shape()[0], x.shape()[2], num_heads, x.shape()[3]//num_heads)
                unt_torch = unt_torch.reshape(new_shape)
                reshaped_unt = _C.tensor.Tensor(
                    unt_torch.reshape(-1).tolist(),
                    unt_torch.shape,
                    _C.tensor.DataFormat.FLOAT32,
                    _C.tensor.Layout.ROW_MAJOR,
                    device
                )
            else:
                #TODO(AP): this reshape doesn't work, need a re-banking reshape here
                #_C.tensor.reshape(untilized_x, x.shape()[0], x.shape()[2], num_heads, x.shape()[3] // num_heads)
                reshaped_unt = untilized_x

            # N, 128, 2, 64
            transposed = _C.tensor.transpose_hc_rm(reshaped_unt)
            # N, 2, 128, 64
            retilized = _C.tensor.tilize(transposed)
            return retilized
            """
            host_tensor = torch.tensor(
                reshaped_untilized_x.to(host).data(),
            ).reshape(reshaped_untilized_x.shape())

            # Doing permute on host until Andrei adds his RM transpose CH
            permuted_tensor = host_tensor.permute(0, 2, 1, 3)
            return _C.tensor.tilize(
                _C.tensor.Tensor(
                    permuted_tensor.reshape(-1).tolist(),
                    permuted_tensor.shape,
                    _C.tensor.DataFormat.FLOAT32,
                    _C.tensor.Layout.ROW_MAJOR,
                    device
                )
            )
            """

    def unmake_attention_heads(x):
        if num_heads == 1:
            return x
        else:
            """
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)
            debug_state["context_reshaped"] = context_layer.clone()

            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
            """
            untilized_x = _C.tensor.untilize(x)
            ctx = _C.tensor.transpose_hc_rm(untilized_x)
            ushape = ctx.shape()
            new_shape = (1, ushape[0], ushape[1], ushape[2]*ushape[3])
            host_reshaped = torch.tensor( tt2torch_rm(ctx) ).reshape( new_shape )
            from_dev = _C.tensor.Tensor(
                host_reshaped.reshape(-1).tolist(),
                host_reshaped.shape,
                _C.tensor.DataFormat.FLOAT32,
                _C.tensor.Layout.ROW_MAJOR,
                device
            )
            #set_FR(1)
            retval = _C.tensor.tilize(from_dev)
            return retval

    def multiply_by_sqrt_hidden_dim(x):
        if num_heads == 1:
            return _C.tensor.bcast(
                x,
                reciprocal_of_sqrt_hidden_dim_tensor,
                _C.tensor.BcastOpMath.MUL,
                _C.tensor.BcastOpDim.HW
            )
        else:
            # Until CHW bcast supported
            hidden_dim = x.shape()[3] # TODO(AP): is this right?
            host_data = x.to(host).data()
            vals_list = [el * 1 / math.sqrt(hidden_dim/num_heads) for el in host_data]
            return _C.tensor.Tensor(
                vals_list,
                x.shape(),
                _C.tensor.DataFormat.FLOAT32,
                _C.tensor.Layout.TILE,
                device
            )

    def mha_(activation):
        Q = QProjection(activation)
        K = KProjection(activation)
        V = VProjection(activation)
        #K_T = _C.tensor.transpose(K)

        Q_heads = make_attention_heads(Q)
        K_heads = make_attention_heads(K)
        V_heads = make_attention_heads(V)
        K_T_heads = _C.tensor.transpose(K_heads)

        qkt = _C.tensor.matmul(Q_heads, K_T_heads)
        # Attention scores computation
        attention_score_input = multiply_by_sqrt_hidden_dim(qkt)

        N, C, H, W = attention_score_input.shape()
        new_shape = [N, 1, C*H, W]
        _C.tensor.reshape(attention_score_input, *new_shape)
        attention_scores = softmax(attention_score_input)
        _C.tensor.reshape(attention_scores, N, C, H, W)

        # Apply attention to value matrix
        weighted_activation = _C.tensor.matmul(attention_scores, V_heads)
        return unmake_attention_heads(weighted_activation)

    return mha_

class TtMultiHeadAttentionModel(torch.nn.Module):
    def __init__(self, state_dict, device):
        super().__init__()
        qw = pad_weight(state_dict["bert.encoder.layer.0.attention.self.query.weight"])
        qb = pad_weight(state_dict["bert.encoder.layer.0.attention.self.query.bias"])
        kw = pad_weight(state_dict["bert.encoder.layer.0.attention.self.key.weight"])
        kb = pad_weight(state_dict["bert.encoder.layer.0.attention.self.key.bias"])
        vw = pad_weight(state_dict["bert.encoder.layer.0.attention.self.value.weight"])
        vb = pad_weight(state_dict["bert.encoder.layer.0.attention.self.value.bias"])

        # Hidden dim
        hidden_dim = qw.shape[-1]

        # Tilized
        parameters = [
            tilize_to_list(qw),
            tilize_to_list(qb),
            tilize_to_list(kw),
            tilize_to_list(kb),
            tilize_to_list(vw),
            tilize_to_list(vb)
        ]

        self.mha = mha(*parameters, hidden_dim, 2, device)

    def forward(self, activation):
        result = self.mha(activation)
        return result

class PytorchMultiHeadAttentionModel(torch.nn.Module):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.mha = hugging_face_reference_model.bert.encoder.layer[0].attention.self

        # Disable dropout
        self.mha.eval()

    def forward(self, x):
        result = self.mha(x)[0]
        return result


def run_mha_inference():
    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained("prajjwal1/bert-tiny", torchscript=False)
    tt_mha_model = TtMultiHeadAttentionModel(hugging_face_reference_model.state_dict(), device)
    pytorch_mha_model = PytorchMultiHeadAttentionModel(hugging_face_reference_model)

    # Prepare input
    torch.manual_seed(0)
    mha_input = (torch.rand(1, 1, 128, 128) * 2) - 1

    pytorch_out = pytorch_mha_model(mha_input.squeeze(1)).unsqueeze(1)

    tt_mha_input = tilize_to_list(pad_activation(mha_input))
    tt_mha_input = _C.tensor.Tensor(tt_mha_input, mha_input.shape, _C.tensor.DataFormat.FLOAT32, _C.tensor.Layout.TILE, device)

    tt_out = tt_mha_model(tt_mha_input).to(host)
    tt_out1 = untilize(torch.Tensor(tt_out.data()).reshape(*pytorch_out.shape))

    print_diff_argmax(pytorch_out, tt_out1)

    # assert np.allclose(pytorch_out.detach().numpy(), tt_out.numpy(), 1e-5, 0.17)

if __name__ == "__main__":
    # Initialize the device
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, 0)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()
    #set_FR(0)
    run_mha_inference()
    _C.device.CloseDevice(device)
