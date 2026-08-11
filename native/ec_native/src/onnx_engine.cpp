#include "onnx_engine.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace ec_native {
namespace {

void require_tensor(const Ort::TypeInfo& type, std::size_t width,
                    const char* kind) {
  const auto tensor = type.GetTensorTypeAndShapeInfo();
  if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error(std::string(kind) +
                             " tensor must use float32 values");
  }
  const auto actual = tensor.GetShape();
  if (actual.size() != 2 || actual[0] != 1 ||
      actual[1] != static_cast<std::int64_t>(width)) {
    throw std::runtime_error(std::string(kind) +
                             " tensor must have static shape [1, " +
                             std::to_string(width) + "]");
  }
}

}  // namespace

OnnxEngine::OnnxEngine(const std::string& model_path, std::string input_name,
                       std::string output_name, std::size_t input_width,
                       std::size_t output_width,
                       std::size_t intra_op_threads)
    : environment_(ORT_LOGGING_LEVEL_WARNING, "ec_native"),
      memory_info_(
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      input_name_(std::move(input_name)),
      output_name_(std::move(output_name)),
      input_names_{input_name_.c_str()},
      output_names_{output_name_.c_str()},
      input_buffer_(input_width, 0.0F),
      output_buffer_(output_width, 0.0F),
      input_shape_{1, static_cast<std::int64_t>(input_width)},
      output_shape_{1, static_cast<std::int64_t>(output_width)} {
  if (intra_op_threads == 0) {
    throw std::runtime_error("ONNX intra-op thread count must be positive");
  }
  session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options_.SetIntraOpNumThreads(static_cast<int>(intra_op_threads));
  session_options_.SetInterOpNumThreads(1);
  session_ = Ort::Session(environment_, model_path.c_str(), session_options_);
  validate_contract();
  input_tensor_ = Ort::Value::CreateTensor<float>(
      memory_info_, input_buffer_.data(), input_buffer_.size(),
      input_shape_.data(), input_shape_.size());
  output_tensor_ = Ort::Value::CreateTensor<float>(
      memory_info_, output_buffer_.data(), output_buffer_.size(),
      output_shape_.data(), output_shape_.size());
}

void OnnxEngine::validate_contract() {
  if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 1) {
    throw std::runtime_error(
        "low-level ONNX model must have exactly one input and one output");
  }
  Ort::AllocatorWithDefaultOptions allocator;
  const auto actual_input = session_.GetInputNameAllocated(0, allocator);
  const auto actual_output = session_.GetOutputNameAllocated(0, allocator);
  if (input_name_ != actual_input.get()) {
    throw std::runtime_error("ONNX input name mismatch: expected " +
                             input_name_ + ", got " + actual_input.get());
  }
  if (output_name_ != actual_output.get()) {
    throw std::runtime_error("ONNX output name mismatch: expected " +
                             output_name_ + ", got " + actual_output.get());
  }
  require_tensor(session_.GetInputTypeInfo(0), input_buffer_.size(), "input");
  require_tensor(session_.GetOutputTypeInfo(0), output_buffer_.size(),
                 "output");
}

std::span<const float> OnnxEngine::infer(std::span<const float> input) {
  if (input.size() != input_buffer_.size()) {
    throw std::runtime_error("ONNX input width mismatch");
  }
  std::copy(input.begin(), input.end(), input_buffer_.begin());
  session_.Run(Ort::RunOptions{nullptr}, input_names_.data(), &input_tensor_,
               1, output_names_.data(), &output_tensor_, 1);
  return output_buffer_;
}

void OnnxEngine::warmup(std::size_t iterations) {
  for (std::size_t index = 0; index < iterations; ++index) {
    infer(input_buffer_);
  }
}

}  // namespace ec_native
