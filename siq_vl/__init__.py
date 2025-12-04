from siq_vl.collator import SiQ_VLDataCollator
from siq_vl.model.configuration import SiQ_VLConfig
from siq_vl.model.modeling import SiQ_VLForCausalLM
from siq_vl.model.processing import SiQ_VLProcessor

__all__ = [
    "SiQ_VLConfig",
    "SiQ_VLDataCollator",
    "SiQ_VLForCausalLM",
    "SiQ_VLProcessor",
]

# Register model and config with Auto classes
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("siq_vl", SiQ_VLConfig)
AutoModelForCausalLM.register(SiQ_VLConfig, SiQ_VLForCausalLM)
