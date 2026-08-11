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

class OnnxEngine {
 public:
  OnnxEngine(const std::string& model_path, std::string input_name,
             std::string output_name, std::size_t input_width,
             std::size_t output_width, std::size_t intra_op_threads = 1);

  OnnxEngine(const OnnxEngine&) = delete;
  OnnxEngine& operator=(const OnnxEngine&) = delete;

  std::span<const float> infer(std::span<const float> input);
  void warmup(std::size_t iterations = 8);

  std::size_t input_width() const noexcept { return input_buffer_.size(); }
  std::size_t output_width() const noexcept { return output_buffer_.size(); }

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
};

}  // namespace ec_native
