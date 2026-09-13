#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace ec_native {

// Where a batch-1 policy / encoder graph runs. "cpu" is the reference path
// (bit-exact with the export parity check). "cuda" appends the CUDA
// execution provider, "tensorrt" the TensorRT provider with CUDA as its
// fallback; both keep host-side input / output buffers, ONNX Runtime does
// the device copies. Measured on the deployment host (RTX 5080, ORT 1.28):
// policy 996 -> 29 at 1.1 ms on one CPU thread, 0.08 ms CUDA, 0.06 ms
// TensorRT fp32; the difference to CUDA graphs (0.045 ms) is not worth the
// device-resident I/O they require. "cuda" reproduces the golden trace to
// 1e-6 like the CPU path; "tensorrt" builds its engines with TF32 matmuls
// (no ONNX Runtime option turns that off) and lands 4e-4 to 8e-4 off on the
// encoder, outside the bundle's parity tolerance, so it is opt-in only.
struct InferenceOptions {
  std::string provider = "cpu";  // cpu | cuda | tensorrt
  int device_id = 0;
  std::size_t intra_op_threads = 1;
  bool trt_fp16 = false;
  // TensorRT builds an engine per graph (2-5 s); a cache directory makes the
  // second session start instant. Empty: rebuilt on every start.
  std::string trt_cache_dir;
};

class OnnxEngine {
 public:
  OnnxEngine(const std::string& model_path, std::string input_name,
             std::string output_name, std::size_t input_width,
             std::size_t output_width, std::size_t intra_op_threads = 1);
  OnnxEngine(const std::string& model_path, std::string input_name,
             std::string output_name, std::size_t input_width,
             std::size_t output_width, const InferenceOptions& options);

  OnnxEngine(const OnnxEngine&) = delete;
  OnnxEngine& operator=(const OnnxEngine&) = delete;

  std::span<const float> infer(std::span<const float> input);
  void warmup(std::size_t iterations = 8);

  std::size_t input_width() const noexcept { return input_buffer_.size(); }
  std::size_t output_width() const noexcept { return output_buffer_.size(); }
  // The provider the session actually runs on ("cpu", "cuda", "tensorrt").
  const std::string& provider() const noexcept { return provider_; }

 private:
  void validate_contract();

  Ort::Env environment_;
  Ort::SessionOptions session_options_;
  Ort::Session session_{nullptr};
  Ort::MemoryInfo memory_info_;
  std::string input_name_;
  std::string output_name_;
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
  std::vector<float> input_buffer_;
  std::vector<float> output_buffer_;
  std::array<std::int64_t, 2> input_shape_{};
  std::array<std::int64_t, 2> output_shape_{};
  Ort::Value input_tensor_{nullptr};
  Ort::Value output_tensor_{nullptr};
  std::string provider_ = "cpu";
};

}  // namespace ec_native
