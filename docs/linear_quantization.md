# Key Features

## Linear Quantization

DAX provides powerful quantization support for linear layers, including various quantization dtype(FP8/INT8), various quantization granularity(per_tensor_symmetric/per_token_symmetric/per_channel_symmetric) and both static and dynamic quantization. Furthermore, DAX allows users to set different quantization strategies for different layers through `qconfig_dict`. Layers without specified qconfig will use its parent's qconfig.

You only need to call a single function to quantization linear layers of your model. By setting the specific layer prefix to `None` in `qconfig_dict`, DAX will skip quantization for these layers. For example, we set "condition_embedder" and "proj_out" to skip quantization for inputs and outputs layers in wan2.1 to achieve better accuracy-performance trade-off.

```python
from dax.quant.quantization import quantize_dynamic
from dax.quant.quantization.qconfig import (
    get_dynamic_int8_per_token_act_per_channel_weight_qconfig,
    get_dynamic_fp8_per_token_act_per_channel_weight_qconfig,
)

qconfig_dict = {
    "": get_dynamic_int8_per_token_act_per_channel_weight_qconfig(),
    # for fp8 linear
    # "": get_dynamic_fp8_per_token_act_per_channel_weight_qconfig(),
    "condition_embedder": None,
    "proj_out": None,
}

quantize_dynamic(model, qconfig_dict)
```


### Linear Quantization Scheme for Wan2.1

For Wan2.1 linear quantization, we use dynamic per-token int8 quantization for activation and static per-channel int8 quantization for weight. Our experiments show that this scheme is the best for linear quantization, boosting the performance by 20% without visible quality loss.

#### quantization error of generated outputs for different activation quantization schemes

| dtype              | Metric           | Latent | Video |
|--------------------|------------------|--------------|-------------|
| pen-token fp8_e4m3 | Cosine Similarity↑ | 0.8635       | 0.8919      |
|                    | MSE↓             | 0.28098      | 0.10663     |
|                    | SNR↓             | 0.28997      | 0.22476     |
| per-token fp8_e5m2 | Cosine Similarity↑ | 0.8629       | 0.8948      |
|                    | MSE↓             | 0.28128      | 0.10317     |
|                    | SNR↓             | 0.29028      | 0.21748     |
| per-token int8     | Cosine Similarity↑ | **0.9759**       | **0.9825**      |
|                    | MSE↓             | **0.04677**      | **0.01658**     |
|                    | SNR↓             | **0.04827**      | **0.03496**     |
| per-tensor int8    | Cosine Similarity↑ | 0.8201       | 0.8529      |
|                    | MSE↓             | 0.3578       | 0.1407      |
|                    | SNR↓             | 0.3692       | 0.2967      |
