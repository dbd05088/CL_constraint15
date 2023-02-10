'''
Copyright (C) 2019-2021 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import sys

import torch.nn as nn

from .pytorch_engine import get_flops_pytorch
from .utils import flops_to_string, params_to_string


def get_model_complexity_info(model, input_res,
                              print_per_layer_stat=True,
                              as_strings=True,
                              input_constructor=None, ost=sys.stdout,
                              verbose=False, ignore_modules=[],
                              custom_modules_hooks={}, backend='pytorch',
                              flops_units=None, param_units=None,
                              output_precision=2, criterion=None, original_opt=None, opt_name=None, lr=None):
    assert type(input_res) is tuple
    assert len(input_res) >= 1
    assert isinstance(model, nn.Module)

    if backend == 'pytorch':
        total_inform, initial_inform, group1_inform, group2_inform, group3_inform, group4_inform, fc_inform = get_flops_pytorch(model, input_res,
                                                      print_per_layer_stat,
                                                      input_constructor, ost,
                                                      verbose, ignore_modules,
                                                      custom_modules_hooks,
                                                      output_precision=output_precision,
                                                      flops_units=flops_units,
                                                      param_units=param_units)

        forward_flops_count, backward_flops_count, params_count, fc_params_count, buffers_count = total_inform
        initial_forward_flops_count, initial_backward_flops_count, initial_params_count = initial_inform
        group1_forward_flops_count, group1_backward_flops_count, group1_params_count = group1_inform
        group2_forward_flops_count, group2_backward_flops_count, group2_params_count = group2_inform
        group3_forward_flops_count, group3_backward_flops_count, group3_params_count = group3_inform
        group4_forward_flops_count, group4_backward_flops_count, group4_params_count = group4_inform
        fc_forward_flops_count, fc_backward_flops_count, fc_params_count = fc_inform


        '''
        flops_count, params_count = get_backward_flops_pytorch(model, input_res,
                                                      print_per_layer_stat,
                                                      input_constructor, ost,
                                                      verbose, ignore_modules,
                                                      custom_modules_hooks,
                                                      output_precision=output_precision,
                                                      flops_units=flops_units,
                                                      param_units=param_units,
                                                      criterion=criterion,
                                                      original_opt=original_opt, 
                                                      opt_name=opt_name, 
                                                      lr=lr)
        '''
    else:
        raise ValueError('Wrong backend name')

    def make_string(forward_flops_count, backward_flops_count, params_count, fc_params_count=None, buffers_count=None):
        forward_flops_string = flops_to_string(
            forward_flops_count,
            units=flops_units,
            precision=output_precision
        )
        backward_flops_string = flops_to_string(
            backward_flops_count,
            units=flops_units,
            precision=output_precision
        )
        params_string = params_to_string(
            params_count,
            units=param_units,
            precision=output_precision
        )
        if fc_params_count is not None:
            fc_params_string = params_to_string(
                fc_params_count,
                units=param_units,
                precision=output_precision
            )
        else:
            fc_params_string = None

        if buffers_count is not None:
            buffers_string = params_to_string(
                buffers_count,
                units=param_units,
                precision=output_precision
            )
        else:
            buffers_string = None

        return forward_flops_string, backward_flops_string, params_string, fc_params_string, buffers_string

    if as_strings:
        forward_flops_string, backward_flops_string, params_string, fc_params_string, buffers_string = make_string(forward_flops_count, backward_flops_count, params_count, fc_params_count, buffers_count)
        initial_forward_flops_string, initial_backward_flops_string, initial_params_string, _, _ = make_string(initial_forward_flops_count, initial_backward_flops_count, initial_params_count)
        group1_forward_flops_string, group1_backward_flops_string, group1_params_string, _, _ = make_string(group1_forward_flops_count, group1_backward_flops_count, group1_params_count)
        group2_forward_flops_string, group2_backward_flops_string, group2_params_string, _, _ = make_string(group2_forward_flops_count, group2_backward_flops_count, group2_params_count)
        group3_forward_flops_string, group3_backward_flops_string, group3_params_string, _, _ = make_string(group3_forward_flops_count, group3_backward_flops_count, group3_params_count)
        group4_forward_flops_string, group4_backward_flops_string, group4_params_string, _, _ = make_string(group4_forward_flops_count, group4_backward_flops_count, group4_params_count)
        fc_forward_flops_string, fc_backward_flops_string, fc_params_string, _, _ = make_string(fc_forward_flops_count, fc_backward_flops_count, fc_params_count)

        return [forward_flops_string, backward_flops_string, params_string, fc_params_string, buffers_string], \
            [initial_forward_flops_string, initial_backward_flops_string, initial_params_string], \
            [group1_forward_flops_string, group1_backward_flops_string, group1_params_string], \
            [group2_forward_flops_string, group2_backward_flops_string, group2_params_string], \
            [group3_forward_flops_string, group3_backward_flops_string, group3_params_string], \
            [group4_forward_flops_string, group4_backward_flops_string, group4_params_string], \
            [fc_forward_flops_string, fc_backward_flops_string, fc_params_string]
        

    return [forward_flops_count, backward_flops_count, params_count, fc_params_count, buffers_count], \
        [initial_forward_flops_count, initial_backward_flops_count, initial_params_count], \
        [group1_forward_flops_count, group1_backward_flops_count, group1_params_count], \
        [group2_forward_flops_count, group2_backward_flops_count, group2_params_count], \
        [group3_forward_flops_count, group3_backward_flops_count, group3_params_count], \
        [group4_forward_flops_count, group4_backward_flops_count, group4_params_count], \
        [fc_forward_flops_count, fc_backward_flops_count, fc_params_count], \
