# CLIP linear approximation for SDXL model based on quantization errors in linear weights

original idea [Malinovskii et al.](https://arxiv.org/abs/2411.17525) is to approximate perplexity based on errors in weights after dequantization:

$$\text{PPL}(\text{Q(model)})\approx\text{PPL}(\text{model})+\sum\limits_{l=1}^{\text{nlayers}}\alpha_l\dfrac{\|W_l-Q(W_l)\|^2_2}{\|W_l\|^2_2}$$

our idea is to take a similar approach to the SDXL (diffusion) model and the CLIP metric:
1) take Hadamard transform of raw weights (first appears in the [Tseng et al.](https://arxiv.org/abs/2402.04396))
2) apply [HQQ](https://mobiusml.github.io/hqq_blog/) quantization based on $L_{p<1}$ minimization of weight quantization error
3) apply dequantization and inverse Hadamard transform (for obtaining quantization errors)
4) calculate CLIP metric for the quantized model
5) analyse the CLIP approximation

- all the essential code in [q_utils](q_utils.py)
- The pipeline $1\rightarrow 3$ is shown in [QuantDequant](QuantDequant.ipynb)
- Final CLIP analysis is shown in [ClipAnalysis](ClipAnalysis.ipynb)
