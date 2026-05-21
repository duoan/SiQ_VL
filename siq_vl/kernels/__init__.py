from siq_vl.kernels.flash_attention import flash_attention, flash_attention_varlen

try:
    from siq_vl.kernels.cutile_attention import (
        cutile_attention,
        cutile_attention_varlen,
    )
except ImportError:
    cutile_attention = None
    cutile_attention_varlen = None
