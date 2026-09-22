#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================================
// gemm_FP8_d.cu - FP8 (e4m3fn) weight-quantized GEMM
// ============================================================================
// Drop-in sibling of gemm_d.cu: y[...,N] = x[...,K] @ Wq[N,K]^T (x scale) (+bias)
//
// Weights are quantized to FP8 E4M3FN with a per-row (output-channel) scale:
//     scale[n] = max|W[n,:]| / 448.0          (448 = max finite e4m3)
//     Wq[n,k]  = e4m3(round(W[n,k] / scale[n]))   (stored as raw uint8 bytes)
//
// The GEMM kernel dequantizes on the fly (byte -> fp32, x scale[n]) and
// accumulates in fp32, so the compute path is W8A16 with full-precision
// accumulation. Weight memory footprint drops to 50% of fp16.
//
// Layout verified against torch.finfo(torch.float8_e4m3fn) and the byte
// patterns produced by torch's own fp16->e4m3fn cast (see benchmarks
// /test_fp8_quant.py): bias 7, exponent field 0..15 all valid, max 0x7E=448,
// subnormal step 2^-9.
// ============================================================================

namespace gemm_fp8 {

// ---------------- e4m3fn <-> float -----------------------------------------

__device__ __forceinline__ float e4m3_to_f32(uint8_t b) {
  int sign = (b >> 7) & 1;
  int e = (b >> 3) & 0xF;
  int m = b & 0x7;
  float v = (e == 0) ? (float)m * 0x1p-9f
                     : ldexpf(1.0f + (float)m * 0.125f, e - 7);
  return sign ? -v : v;
}

__device__ __forceinline__ uint8_t fp32_to_e4m3(float v) {
  float av = fabsf(v);
  int sign = (v < 0.0f) ? 0x80 : 0;
  if (av == 0.0f) return (uint8_t)sign;
  if (av >= 448.0f) return (uint8_t)(sign | 0x7E);  // saturate to max

  uint32_t i = __float_as_int(av);
  int e8 = (int)((i >> 23) & 0xFF) - 127;  // fp32 unbiased exponent
  int m23 = (int)(i & 0x7FFFFF);

  if (e8 < -10) return (uint8_t)sign;               // flush to zero (< 2^-10)
  if (e8 < -6) {                                    // subnormal domain (2^-10..2^-6)
    int sub = (int)rintf(av * 512.0f);
    if (sub > 7) sub = 7;
    return (uint8_t)(sign | (uint8_t)sub);
  }

  int m3 = (m23 + 0x80000) >> 20;                   // round-to-nearest at 3 mantissa bits
  if (m3 == 8) { m3 = 0; e8 += 1; }                 // mantissa carry
  int e = e8 + 7;
  if (e > 15) { return (uint8_t)(sign | 0x7E); }   // saturate (>= 448)
  if (m3 > 6 && e == 15) m3 = 6;                   // 0x7F is NaN; clamp to 448
  return (uint8_t)(sign | (uint8_t)(m3 | (e << 3)));
}

// ---------------- quantization kernel --------------------------------------

__global__ void quantize_kernel(const __half* __restrict__ w,
                                uint8_t* __restrict__ wq,
                                float* __restrict__ scales,
                                long N, long K) {
  long n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;

  float maxv = 0.0f;
  for (long k = 0; k < K; ++k) {
    float v = __half2float(w[n * K + k]);
    maxv = fmaxf(maxv, fabsf(v));
  }
  float scale = (maxv > 0.0f) ? maxv / 448.0f : 1.0f;
  float inv = 1.0f / scale;

  for (long k = 0; k < K; ++k) {
    float v = __half2float(w[n * K + k]);
    wq[n * K + k] = fp32_to_e4m3(v * inv);   // encoder rounds to the e4m3 grid
  }
  scales[n] = scale;
}

// ---------------- FP8 GEMM (dequantizing, fp32 accumulation) ---------------

constexpr int F8_BM = 8;
constexpr int F8_BN = 64;
constexpr int F8_BK = 16;

__global__ void fp8_gemm_kernel(const __half* __restrict__ x,
                                const uint8_t* __restrict__ wq,
                                const float* __restrict__ scales,
                                const __half* __restrict__ bias,
                                __half* __restrict__ y,
                                long M, long N, long K) {
  __shared__ __half s_x[F8_BM][F8_BK];
  __shared__ __half s_w[F8_BK][F8_BN];
  __shared__ float s_sc[F8_BN];

  long mt = (long)blockIdx.y * F8_BM;
  long nt = (long)blockIdx.x * F8_BN;
  int tid = threadIdx.x;
  int m  = tid / F8_BN;
  int n  = tid % F8_BN;

  if (tid < F8_BN) {
    s_sc[tid] = (nt + tid < N) ? scales[nt + tid] : 1.0f;
  }
  __syncthreads();

  float acc = 0.0f;
  for (long k0 = 0; k0 < K; k0 += F8_BK) {
    for (int i = tid; i < F8_BM * F8_BK; i += blockDim.x) {
      int r = i / F8_BK, c = i % F8_BK;
      long gk = k0 + c;
      s_x[r][c] = (mt + r < M && gk < K) ? x[(mt + r) * K + gk] : __float2half(0.0f);
    }
    for (int i = tid; i < F8_BK * F8_BN; i += blockDim.x) {
      int r = i / F8_BN, c = i % F8_BN;
      long gk = k0 + r;
      float wv = 0.0f;
      if (nt + c < N && gk < K) {
        wv = e4m3_to_f32(wq[(nt + c) * K + gk]) * s_sc[c];
      }
      s_w[r][c] = __float2half(wv);
    }
    __syncthreads();

    if (mt + m < M && nt + n < N) {
      #pragma unroll
      for (int k = 0; k < F8_BK; ++k) {
        acc += __half2float(s_x[m][k]) * __half2float(s_w[k][n]);
      }
    }
    __syncthreads();
  }

  if (mt + m < M && nt + n < N) {
    if (bias) acc += __half2float(bias[nt + n]);
    y[(mt + m) * N + nt + n] = __float2half(acc);
  }
}

// ---------------- quantization: [N,K] fp16 -> (uint8 e4m3, fp32 scale) -----

std::vector<torch::Tensor> quantize(torch::Tensor w) {
  TORCH_CHECK(w.is_cuda() && w.scalar_type() == at::kHalf, "gemm_fp8: fp16 CUDA weight required");
  TORCH_CHECK(w.dim() == 2, "gemm_fp8: weight must be [N,K]");
  auto wc = w.contiguous();
  long N = wc.size(0), K = wc.size(1);

  auto wq = torch::empty({N, K}, wc.options().dtype(at::kByte));
  auto scales = torch::empty({N}, wc.options().dtype(at::kFloat));

  int threads = 128;
  int blocks = (int)((N + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  quantize_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __half*>(wc.data_ptr<at::Half>()),
      wq.data_ptr<uint8_t>(), scales.data_ptr<float>(), N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {wq, scales};
}

// ---------------- FP8 GEMM on pre-quantized weights ------------------------

torch::Tensor f8_forward(torch::Tensor x, torch::Tensor wq, torch::Tensor scales,
                         c10::optional<torch::Tensor> bias) {
  TORCH_CHECK(x.is_cuda() && wq.is_cuda(), "gemm_fp8: CUDA tensors required");
  TORCH_CHECK(x.scalar_type() == at::kHalf, "gemm_fp8: fp16 x required");
  TORCH_CHECK(wq.scalar_type() == at::kByte, "gemm_fp8: wq must be uint8 e4m3");
  TORCH_CHECK(wq.dim() == 2 && wq.size(0) == scales.numel(), "gemm_fp8: shape mismatch");

  const long N = wq.size(0);
  const long K = wq.size(1);
  TORCH_CHECK(x.size(-1) == K, "gemm_fp8: inner dim mismatch");

  auto xc = x.contiguous();
  auto x2 = xc.reshape({-1, K});
  const long M = x2.size(0);

  auto y2 = torch::empty({M, N}, x.options());

  const __half* bptr = nullptr;
  if (bias.has_value() && bias->defined() && bias->numel() > 0) {
    TORCH_CHECK(bias->numel() == N, "gemm_fp8: bias length mismatch");
    bptr = reinterpret_cast<const __half*>(bias->contiguous().data_ptr<at::Half>());
  }

  dim3 block(F8_BM * F8_BN);
  dim3 grid((int)((N + F8_BN - 1) / F8_BN), (int)((M + F8_BM - 1) / F8_BM));
  auto stream = at::cuda::getCurrentCUDAStream();

  fp8_gemm_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __half*>(x2.data_ptr<at::Half>()),
      wq.data_ptr<uint8_t>(),
      scales.data_ptr<float>(),
      bptr,
      reinterpret_cast<__half*>(y2.data_ptr<at::Half>()),
      M, N, K);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  std::vector<int64_t> out_shape(x.sizes().begin(), x.sizes().end() - 1);
  out_shape.push_back(N);
  return y2.reshape(out_shape);
}

// ---------------- drop-in: quantize + GEMM ------------------------------

torch::Tensor forward(torch::Tensor x, torch::Tensor w, c10::optional<torch::Tensor> bias) {
  auto q = quantize(w);
  return f8_forward(x, q[0], q[1], bias);
}

}  // namespace gemm_fp8

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gemm_fp8::forward, "FP8 e4m3 weight-quantized linear y[...,N] = x[...,K] @ Wq[N,K]^T (+bias)",
        py::arg("x"), py::arg("w"), py::arg("bias") = py::none());
  m.def("quantize", &gemm_fp8::quantize, "quantize fp16 [N,K] weights to e4m3fn + per-row fp32 scales",
        py::arg("w"));
  m.def("f8_forward", &gemm_fp8::f8_forward, "FP8 GEMM on pre-quantized weights",
        py::arg("x"), py::arg("wq"), py::arg("scales"), py::arg("bias") = py::none());
}