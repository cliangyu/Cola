from transformers.models.t5.modeling_t5 import (
    PARALLELIZE_DOCSTRING,
    add_start_docstrings,
    T5ForConditionalGeneration,
)
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map
import torch


class T5OnSpecificDevices(T5ForConditionalGeneration):
    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None, device_ids=range(torch.cuda.device_count())):
        self.device_map = (
            get_device_map(len(self.encoder.block), device_ids)  # assign all GPUs
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True
