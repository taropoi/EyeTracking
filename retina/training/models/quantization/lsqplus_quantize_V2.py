import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from .lsqquantize_V1 import Round


class ALSQPlus(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, beta):
        # assert alpha > 0, "alpha={}".format(alpha)
        ctx.save_for_backward(weight, alpha, beta)
        ctx.other = g, Qn, Qp
        w_q = Round.apply(torch.div((weight - beta), alpha).clamp(Qn, Qp))
        w_q = w_q * alpha + beta
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha, beta = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        q_w = (weight - beta) / alpha
        smaller = (q_w < Qn).float()  # bool值转浮点值，1.0或者0.0
        bigger = (q_w > Qp).float()  # bool值转浮点值，1.0或者0.0
        between = 1.0 - smaller - bigger  # 得到位于量化区间的index
        grad_alpha = (
            (
                (
                    smaller * Qn
                    + bigger * Qp
                    + between * Round.apply(q_w)
                    - between * q_w
                )
                * grad_weight
                * g
            )
            .sum()
            .unsqueeze(dim=0)
        )
        grad_beta = ((smaller + bigger) * grad_weight * g).sum().unsqueeze(dim=0)
        # 在量化区间之外的值都是常数，故导数也是0
        grad_weight = between * grad_weight
        # 返回的梯度要和forward的参数对应起来
        return grad_weight, grad_alpha, None, None, None, grad_beta


class WLSQPlus(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, per_channel):
        # assert alpha > 0, "alpha={}".format(alpha)
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp, per_channel
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            w_q = Round.apply(torch.div(weight, alpha).clamp(Qn, Qp))
            w_q = w_q * alpha
            w_q = torch.transpose(w_q, 0, 1)
            w_q = w_q.contiguous().view(sizes)
        else:
            w_q = Round.apply(torch.div(weight, alpha).clamp(Qn, Qp))
            w_q = w_q * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp, per_channel = ctx.other
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            q_w = weight / alpha
            q_w = torch.transpose(q_w, 0, 1)
            q_w = q_w.contiguous().view(sizes)
        else:
            q_w = weight / alpha
        smaller = (q_w < Qn).float()  # bool值转浮点值，1.0或者0.0
        bigger = (q_w > Qp).float()  # bool值转浮点值，1.0或者0.0
        between = 1.0 - smaller - bigger  # 得到位于量化区间的index
        if per_channel:
            grad_alpha = (
                (
                    smaller * Qn
                    + bigger * Qp
                    + between * Round.apply(q_w)
                    - between * q_w
                )
                * grad_weight
                * g
            )
            grad_alpha = (
                grad_alpha.contiguous().view(grad_alpha.size()[0], -1).sum(dim=1)
            )
        else:
            grad_alpha = (
                (
                    (
                        smaller * Qn
                        + bigger * Qp
                        + between * Round.apply(q_w)
                        - between * q_w
                    )
                    * grad_weight
                    * g
                )
                .sum()
                .unsqueeze(dim=0)
            )
        # 在量化区间之外的值都是常数，故导数也是0
        grad_weight = between * grad_weight
        return grad_weight, grad_alpha, None, None, None, None


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


def get_percentile_min_max(input, lower_percentile, uppper_percentile, output_tensor):
    batch_size = input.shape[0]
    lower_index = round(batch_size * (1 - lower_percentile * 0.01))
    upper_index = round(batch_size * (1 - uppper_percentile * 0.01))

    upper_bound = torch.kthvalue(input, k=upper_index).values

    if lower_percentile == 0:
        lower_bound = upper_bound * 0
    else:
        low_bound = -torch.kthvalue(-input, k=lower_index).values


# def update_scale_betas():
#     for m in model.modules():
#         if isinstance(m, nn.)


# A(特征)量化
class LSQPlusActivationQuantizer(nn.Module):
    def __init__(self, a_bits, all_positive=False, batch_init=20):
        # activations 没有per-channel这个选项的
        super(LSQPlusActivationQuantizer, self).__init__()
        self.a_bits = a_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2**self.a_bits - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = -(2 ** (self.a_bits - 1))
            self.Qp = 2 ** (self.a_bits - 1) - 1
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.ones(0), requires_grad=True)
        self.init_state = 0

    # 量化/反量化
    def forward(self, activation):
        if self.init_state == 0:
            self.g = 1.0 / math.sqrt(activation.numel() * self.Qp)
            mina = torch.min(activation.detach())
            self.s.data = (torch.max(activation.detach()) - mina) / (self.Qp - self.Qn)
            self.beta.data = mina - self.s.data * self.Qn
            self.init_state += 1
        elif self.init_state < self.batch_init:
            mina = torch.min(activation.detach())
            self.s.data = self.s.data * 0.9 + 0.1 * (
                torch.max(activation.detach()) - mina
            ) / (self.Qp - self.Qn)
            self.beta.data = self.s.data * 0.9 + 0.1 * (mina - self.s.data * self.Qn)
            self.init_state += 1
        elif self.init_state == self.batch_init:
            # self.s = torch.nn.Parameter(self.s)
            # self.beta = torch.nn.Parameter(self.beta)
            self.init_state += 1

        if self.a_bits == 32:
            q_a = activation
        elif self.a_bits == 1:
            print("！Binary quantization is not supported ！")
            assert self.a_bits != 1
        else:
            q_a = ALSQPlus.apply(
                activation, self.s, self.g, self.Qn, self.Qp, self.beta
            )
        return q_a


# W(权重)量化
class LSQPlusWeightQuantizer(nn.Module):
    def __init__(self, w_bits, all_positive=False, per_channel=False, batch_init=20):
        super(LSQPlusWeightQuantizer, self).__init__()
        self.w_bits = w_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            # unsigned activation is quantized to [0, 2^b-1]
            self.Qn = 0
            self.Qp = 2**w_bits - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.Qn = -(2 ** (w_bits - 1))
            self.Qp = 2 ** (w_bits - 1) - 1
        self.per_channel = per_channel
        self.init_state = 0
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    # 量化/反量化
    def forward(self, weight):
        """
                For this work, each layer of weights and each layer of activations has a distinct step size, represented
        as an fp32 value, initialized to 2h|v|i/√OP , computed on either the initial weights values or the first
        batch of activations, respectively
        """
        if self.init_state == 0:
            self.g = 1.0 / math.sqrt(weight.numel() * self.Qp)
            self.div = 2**self.w_bits - 1
            if self.per_channel:
                weight_tmp = weight.detach().contiguous().view(weight.size()[0], -1)
                mean = torch.mean(weight_tmp, dim=1)
                std = torch.std(weight_tmp, dim=1)
                self.s.data, _ = torch.max(
                    torch.stack([torch.abs(mean - 3 * std), torch.abs(mean + 3 * std)]),
                    dim=0,
                )
                self.s.data = self.s.data / self.div
            else:
                mean = torch.mean(weight.detach())
                std = torch.std(weight.detach())
                self.s.data = (
                    max([torch.abs(mean - 3 * std), torch.abs(mean + 3 * std)])
                    / self.div
                )
            self.init_state += 1
        elif self.init_state < self.batch_init:
            self.div = 2**self.w_bits - 1
            if self.per_channel:
                weight_tmp = weight.detach().contiguous().view(weight.size()[0], -1)
                mean = torch.mean(weight_tmp, dim=1)
                std = torch.std(weight_tmp, dim=1)
                self.s.data, _ = torch.max(
                    torch.stack([torch.abs(mean - 3 * std), torch.abs(mean + 3 * std)]),
                    dim=0,
                )
                self.s.data = self.s.data * 0.9 + 0.1 * self.s.data / self.div
            else:
                mean = torch.mean(weight.detach())
                std = torch.std(weight.detach())
                self.s.data = (
                    self.s.data * 0.9
                    + 0.1
                    * max([torch.abs(mean - 3 * std), torch.abs(mean + 3 * std)])
                    / self.div
                )
            self.init_state += 1
        elif self.init_state == self.batch_init:
            # self.s = torch.nn.Parameter(self.s)
            self.init_state += 1

        if self.w_bits == 32:
            output = weight
        elif self.w_bits == 1:
            print("！Binary quantization is not supported ！")
            assert self.w_bits != 1
        else:
            w_q = WLSQPlus.apply(
                weight, self.s, self.g, self.Qn, self.Qp, self.per_channel
            )

            # alpha = grad_scale(self.s, g)
            # w_q = Round.apply((weight/alpha).clamp(Qn, Qp)) * alpha
        return w_q


class QuantConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        a_bits=8,
        w_bits=8,
        quant_inference=False,
        all_positive=False,
        per_channel=False,
        batch_init=20,
    ):
        super(QuantConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.quant_inference = quant_inference
        self.activation_quantizer = LSQPlusActivationQuantizer(
            a_bits=a_bits, all_positive=all_positive, batch_init=batch_init
        )
        self.weight_quantizer = LSQPlusWeightQuantizer(
            w_bits=w_bits,
            all_positive=all_positive,
            per_channel=per_channel,
            batch_init=batch_init,
        )

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight

        output = F.conv2d(
            quant_input,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output


class QuantConvTranspose2d(nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        a_bits=8,
        w_bits=8,
        quant_inference=False,
        all_positive=False,
        per_channel=False,
        batch_init=20,
    ):
        super(QuantConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.quant_inference = quant_inference
        self.activation_quantizer = LSQPlusActivationQuantizer(
            a_bits=a_bits, all_positive=all_positive, batch_init=batch_init
        )
        self.weight_quantizer = LSQPlusWeightQuantizer(
            w_bits=w_bits,
            all_positive=all_positive,
            per_channel=per_channel,
            batch_init=batch_init,
        )

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.conv_transpose2d(
            quant_input,
            quant_weight,
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )
        return output


class QuantLinear(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        a_bits=8,
        w_bits=8,
        quant_inference=False,
        all_positive=False,
        per_channel=False,
        batch_init=20,
    ):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_inference = quant_inference
        self.activation_quantizer = LSQPlusActivationQuantizer(
            a_bits=a_bits, all_positive=all_positive, batch_init=batch_init
        )
        self.weight_quantizer = LSQPlusWeightQuantizer(
            w_bits=w_bits,
            all_positive=all_positive,
            per_channel=per_channel,
            batch_init=batch_init,
        )

    def forward(self, input):
        quant_input = self.activation_quantizer(input)
        if not self.quant_inference:
            quant_weight = self.weight_quantizer(self.weight)
        else:
            quant_weight = self.weight
        output = F.linear(quant_input, quant_weight, self.bias)
        return output

def count_layers(module):
    layer_count = 0
    for child in module.children():
        if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            layer_count += 1
        else:
            layer_count += count_layers(child)
    return layer_count

def add_quant_op(
    module,
    layer_counter,
    total_layers,
    a_bits=8,
    w_bits=8,
    quant_inference=False,
    all_positive=False,
    per_channel=False,
    batch_init=20,
):
    for name, child in module.named_children():
        if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            layer_counter[0] += 1
            if layer_counter[0] == 1 or layer_counter[0] == total_layers:
                # Skip quantization for the first and last layer
                continue
            
            # Quantize the layer
            if isinstance(child, nn.Conv2d):
                quant_layer = QuantConv2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=(child.bias is not None),
                    padding_mode=child.padding_mode,
                    a_bits=a_bits,
                    w_bits=w_bits,
                    quant_inference=quant_inference,
                    all_positive=all_positive,
                    per_channel=per_channel,
                    batch_init=batch_init,
                )
                if child.bias is not None:
                    quant_layer.bias.data = child.bias
                quant_layer.weight.data = child.weight
                module._modules[name] = quant_layer
                
            elif isinstance(child, nn.ConvTranspose2d):
                quant_layer = QuantConvTranspose2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    output_padding=child.output_padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=(child.bias is not None),
                    padding_mode=child.padding_mode,
                    a_bits=a_bits,
                    w_bits=w_bits,
                    quant_inference=quant_inference,
                    all_positive=all_positive,
                    per_channel=per_channel,
                    batch_init=batch_init,
                )
                if child.bias is not None:
                    quant_layer.bias.data = child.bias
                quant_layer.weight.data = child.weight
                module._modules[name] = quant_layer
                
            elif isinstance(child, nn.Linear):
                quant_layer = QuantLinear(
                    child.in_features,
                    child.out_features,
                    bias=(child.bias is not None),
                    a_bits=a_bits,
                    w_bits=w_bits,
                    quant_inference=quant_inference,
                    all_positive=all_positive,
                    per_channel=per_channel,
                    batch_init=batch_init,
                )
                if child.bias is not None:
                    quant_layer.bias.data = child.bias
                quant_layer.weight.data = child.weight
                module._modules[name] = quant_layer
                
        else:
            add_quant_op(
                child,
                layer_counter,
                total_layers,
                a_bits=a_bits,
                w_bits=w_bits,
                quant_inference=quant_inference,
                all_positive=all_positive,
                per_channel=per_channel,
                batch_init=batch_init,
            )



def prepare(
    model,
    inplace=False,
    a_bits=8,
    w_bits=8,
    quant_inference=False,
    all_positive=False,
    per_channel=False,
    batch_init=20,
):

    if not inplace:
        model = copy.deepcopy(model)

    total_layers = count_layers(model)
    layer_counter = [0]
    
    add_quant_op(
        model,
        layer_counter,
        a_bits=a_bits,
        w_bits=w_bits,
        total_layers=total_layers,
        quant_inference=quant_inference,
        all_positive=all_positive,
        per_channel=per_channel,
        batch_init=batch_init,
    )
    return model
