/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   trainer.h
 *  @author Thomas Müller, NVIDIA
 *  @brief  Class that performs training of a differentiable cuda object, given an optimizer and a loss.
 */

#pragma once

#include <memopt-adapter/adapter.h>
#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/common_host.h>
#include <tiny-cuda-nn/cuda_graph.h>
#include <tiny-cuda-nn/gpu_memory_json.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/object.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/random.h>
#include <tiny-cuda-nn/reduce_sum.h>

#include <chrono>
#include <random>

namespace tcnn {

template <typename T, typename PARAMS_T, typename COMPUTE_T = T>
class Trainer : public ObjectWithMutableHyperparams {
 public:
  Trainer(std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> model, std::shared_ptr<Optimizer<PARAMS_T>> optimizer, std::shared_ptr<Loss<COMPUTE_T>> loss, uint32_t seed = 1337, float perturbation_sigma = 0)
      : m_model{model}, m_optimizer{optimizer}, m_loss{loss}, m_perturbation_sigma{perturbation_sigma} {
    std::seed_seq seq{seed};
    std::vector<uint32_t> seeds(2);
    seq.generate(std::begin(seeds), std::end(seeds));
    m_rng = pcg32{seeds.front()};
    initialize_params();
  }

  virtual ~Trainer() {}

  void set_loss(std::shared_ptr<Loss<COMPUTE_T>> loss) {
    if (!loss) {
      throw std::runtime_error{"Trainer: may not set loss to nullptr"};
    }
    m_loss = loss;
  }

  void initialize_params() {
    size_t n_params = m_model->n_params();
    log_debug("Trainer: initializing {} params and resetting training.", n_params);

    // Allocate auxiliary optimizer buffers
    m_optimizer->allocate(m_model);

    m_params_buffer.resize(sizeof(PARAMS_T) * n_params * 2 + sizeof(float) * n_params * 1);
    m_params_buffer.memset(0);

    printf("trainer.m_params_buffer (MiB): %.6lf\n", (double)m_params_buffer.get_bytes() / 1024.0 / 1024.0);

    reset_param_pointers();

    m_model->initialize_params(m_rng, m_params_full_precision);

    // initialize_params is only expected to initialize m_params_full_precision. Cast and copy these over!
    parallel_for_gpu(n_params, [params_fp = m_params_full_precision, params = m_params] __device__(size_t i) {
      params[i] = (PARAMS_T)params_fp[i];
    });
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
  }

  struct ForwardContext : public Context {
    GPUMatrix<COMPUTE_T> perturbed_output;
    GPUMatrix<COMPUTE_T> output;
    GPUMatrix<COMPUTE_T> dL_doutput;
    GPUMatrix<float> L;
    std::unique_ptr<Context> model_ctx;
  };

  std::shared_ptr<ForwardContext> forward_alloc(
    cudaStream_t stream,
    const float loss_scale,
    const GPUMatrixDynamic<T>& input,
    const GPUMatrix<float>& target,
    const GPUMatrix<float>* data_pdf = nullptr,
    bool use_inference_params = false,
    bool prepare_input_gradients = false,
    const GPUMatrix<COMPUTE_T>* external_dL_dy = nullptr
  ) {
    const uint32_t batch_size = input.n();

    auto forward = std::make_shared<ForwardContext>();

    forward->output = GPUMatrix<COMPUTE_T>{m_model->padded_output_width(), batch_size, nullptr, memopt::ConfigurationManager::getConfig().generic.useUM};
    memopt_adapter::register_array(forward->output);

    forward->model_ctx = m_model->forward_alloc(stream, input, &forward->output, use_inference_params, prepare_input_gradients);

    if (m_perturbation_sigma > 0) {
      forward->perturbed_output = GPUMatrix<COMPUTE_T>{m_model->padded_output_width(), batch_size, nullptr, memopt::ConfigurationManager::getConfig().generic.useUM};
      memopt_adapter::register_array(forward->perturbed_output);
    }

    forward->L = GPUMatrix<float>{m_model->padded_output_width(), batch_size, nullptr, memopt::ConfigurationManager::getConfig().generic.useUM};
    memopt_adapter::register_array(forward->L);

    if (external_dL_dy) {
      CHECK_THROW(external_dL_dy->m() == m_model->padded_output_width());
      CHECK_THROW(external_dL_dy->n() == batch_size);

      forward->dL_doutput = GPUMatrix<COMPUTE_T>{external_dL_dy->data(), m_model->padded_output_width(), batch_size};
    } else {
      CHECK_THROW(input.n() == target.n());
      CHECK_THROW(m_model->output_width() == target.m());

      forward->dL_doutput = GPUMatrix<COMPUTE_T>{m_model->padded_output_width(), batch_size, nullptr, memopt::ConfigurationManager::getConfig().generic.useUM};
      memopt_adapter::register_array(forward->dL_doutput);
    }

    return forward;
  }

  void forward(
    cudaStream_t stream,
    ForwardContext& forward,
    const float loss_scale,
    const GPUMatrixDynamic<T>& input,
    const GPUMatrix<float>& target,
    const GPUMatrix<float>* data_pdf = nullptr,
    bool use_inference_params = false,
    bool prepare_input_gradients = false,
    const GPUMatrix<COMPUTE_T>* external_dL_dy = nullptr
  ) {
    const uint32_t batch_size = input.n();

    m_model->forward(stream, *forward.model_ctx, input, &forward.output, use_inference_params, prepare_input_gradients);

    if (m_perturbation_sigma > 0) {
      GPUMatrix<float> perturbation{m_model->padded_output_width(), batch_size, stream};

      const uint32_t n_elements = perturbation.n_elements();
      generate_random_logistic<float>(stream, m_rng, n_elements, perturbation.data(), 0.0f, m_perturbation_sigma);
      add<<<n_blocks_linear(n_elements), N_THREADS_LINEAR, 0, stream>>>(n_elements, forward.output.data(), perturbation.data(), forward.perturbed_output.data());
    }

    auto& loss_input = m_perturbation_sigma > 0 ? forward.perturbed_output : forward.output;

    if (!external_dL_dy) {
      m_loss->evaluate(stream, loss_scale, loss_input, target, forward.L, forward.dL_doutput, data_pdf);
    }
  }

  void backward(cudaStream_t stream, const ForwardContext& ctx, const GPUMatrixDynamic<T>& input, GPUMatrixDynamic<T>* dL_dinput = nullptr, bool use_inference_params = false, GradientMode param_gradients_mode = GradientMode::Overwrite) {
    m_model->backward(stream, *ctx.model_ctx, input, ctx.output, ctx.dL_doutput, dL_dinput, use_inference_params, param_gradients_mode);
  }

  void optimizer_step(cudaStream_t stream, float loss_scale) {
    m_optimizer->step(stream, loss_scale, m_params_full_precision, m_params, m_param_gradients);
  }

  std::shared_ptr<ForwardContext> train(
    uint32_t batch_size,
    uint32_t steps,
    std::function<void(cudaStream_t, uint32_t, GPUMatrixDynamic<T>&, GPUMatrix<float>&)> generateTrainingData,
    cudaStream_t stream,
    const GPUMatrix<float>* data_pdf = nullptr,
    bool run_optimizer = true,
    GPUMatrixDynamic<T>* dL_dinput = nullptr,
    bool use_inference_params = false,
    GradientMode param_gradients_mode = GradientMode::Overwrite,
    const GPUMatrix<COMPUTE_T>* external_dL_dy = nullptr
  ) {
    const float loss_scale = default_loss_scale<PARAMS_T>();

    const uint32_t n_input_dims = m_model->input_width();
    const uint32_t n_output_dims = m_model->output_width();

    // Auxiliary matrices for training
    GPUMatrix<float> input(n_input_dims, batch_size);
    GPUMatrix<float> target(n_output_dims, batch_size);

    m_training_ctx = forward_alloc(stream, loss_scale, input, target, data_pdf, use_inference_params, dL_dinput, external_dL_dy);

    {
      auto capture_guard = m_graph.capture_guard(stream);

      memopt_adapter::Task task = [&](cudaStream_t stream) {
        generateTrainingData(stream, batch_size, input, target);
      };
      memopt_adapter::register_and_execute_task(
        {},
        {},
        task,
        stream
      );

      forward(stream, *m_training_ctx, loss_scale, input, target, data_pdf, use_inference_params, dL_dinput, external_dL_dy);
      backward(stream, *m_training_ctx, input, dL_dinput, use_inference_params, param_gradients_mode);

      if (run_optimizer) {
        optimizer_step(stream, loss_scale);
      }
    }

    CUDA_CHECK_THROW(cudaGraphDebugDotPrint(m_graph.graph(), "graph.dot", cudaGraphDebugDotFlagsVerbose));

    if (memopt::ConfigurationManager::getConfig().generic.optimize) {
      // Start timing the preprocessing (profiling and optimization) phase
      std::chrono::steady_clock::time_point preprocess_start = std::chrono::steady_clock::now();
      
      auto optimized_graph = memopt::profileAndOptimize(m_graph.graph());
      
      std::chrono::steady_clock::time_point preprocess_end = std::chrono::steady_clock::now();
      float preprocessing_time = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start).count() / 1000000.0f;
      printf("FRUGAL preprocessing time (profiling + optimization) (s): %.6f\n", preprocessing_time);

      // TODO: Reset parameters after profiling

      int iterations;
      float running_time;
      std::map<void*, void*> managed_device_array_to_host_array_map;
      memopt::executeOptimizedGraphRepeatedly(
        optimized_graph,
        memopt_adapter::execute_random_task,
        [steps, i = int(0)]() mutable {
          i++;
          return i <= steps;
        },
        iterations,
        running_time,
        managed_device_array_to_host_array_map
      );
      printf("running_time of optimized_graph (s): %.6f\n", running_time);
    } else {
      memopt::PeakMemoryUsageProfiler profiler;

      cudaGraphExec_t graphExec;
      CUDA_CHECK_THROW(cudaGraphInstantiate(&graphExec, m_graph.graph()));
      CUDA_CHECK_THROW(cudaGraphUpload(graphExec, stream));
      CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

      if (memopt::ConfigurationManager::getConfig().generic.useUM) {
        size_t available = 1024ULL * 1024ULL * memopt::ConfigurationManager::getConfig().generic.availableMemoryForUMInMiB;
        memopt::reduceAvailableMemoryForUM(available);

        size_t sum = 0;
        for (auto matrix : memopt_adapter::managedMatrices) {
          if (sum + matrix->n_bytes() > available) break;
          sum += matrix->n_bytes();
          // Skip prefetching to GPU - keep memory on CPU at start for unified memory testing
          // CUDA_CHECK_THROW(cudaMemPrefetchAsync(((GPUMatrixDynamic<COMPUTE_T>*)matrix)->data(), matrix->n_bytes(), memopt::ConfigurationManager::getConfig().execution.mainDeviceId, stream));
          // Optional: Explicitly prefetch to CPU to ensure it's on CPU side
          CUDA_CHECK_THROW(cudaMemPrefetchAsync(((GPUMatrixDynamic<COMPUTE_T>*)matrix)->data(), matrix->n_bytes(), cudaCpuDeviceId, stream));
        }
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
      }

      profiler.start();

      memopt::CudaEventClock clock;
      clock.start(stream);
      for (int i = 0; i < steps; i++) {
        printf("i = %d\n", i);
        CUDA_CHECK_THROW(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
      }
      clock.end(stream);
      CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
      printf("running_time of unoptimized graph (s): %.6f\n", clock.getTimeInSeconds());

      size_t peakMem = profiler.end();
      printf("peak memory usage (MiB): %.6lf\n", (double)peakMem / 1024.0 / 1024.0);

      if (memopt::ConfigurationManager::getConfig().generic.useUM) {
        memopt::resetAvailableMemoryForUM();
      }

      CUDA_CHECK_THROW(cudaGraphExecDestroy(graphExec));
    }

    return m_training_ctx;
  }

  float loss(cudaStream_t stream, const ForwardContext& ctx) const {
    return reduce_sum(ctx.L.data(), ctx.L.n_elements(), stream);
  }

  float loss(const ForwardContext& ctx) const {
    return loss(nullptr, ctx);
  }

  void update_hyperparams(const json& params) override {
    m_optimizer->update_hyperparams(params.value("optimizer", json::object()));
    m_loss->update_hyperparams(params.value("loss", json::object()));
  }

  json hyperparams() const override {
    return {
      {"otype", "Trainer"},
      {"optimizer", m_optimizer->hyperparams()},
      {"loss", m_loss->hyperparams()},
    };
  }

  float* params_full_precision() const {
    return m_params_full_precision;
  }

  PARAMS_T* params() const {
    return m_params;
  }

  PARAMS_T* params_inference() const {
    return m_params_inference;
  }

  PARAMS_T* param_gradients() const {
    return m_param_gradients;
  }

  void set_params_full_precision(const float* params, size_t n_params, bool device_ptr = false) {
    if (n_params != m_model->n_params()) {
      throw std::runtime_error{"Can't set fp params because buffer has the wrong size."};
    }
    CUDA_CHECK_THROW(cudaMemcpy(m_params_full_precision, params, sizeof(float) * n_params, device_ptr ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));

    parallel_for_gpu(n_params, [params_fp = m_params_full_precision, params_inference = m_params_inference] __device__(size_t i) {
      params_inference[i] = (PARAMS_T)params_fp[i];
    });

    CUDA_CHECK_THROW(cudaMemcpy(m_params, m_params_inference, sizeof(PARAMS_T) * n_params, cudaMemcpyDeviceToDevice));
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
  }

  void set_params(const PARAMS_T* params, size_t n_params, bool device_ptr = false) {
    if (n_params != m_model->n_params()) {
      throw std::runtime_error{"Can't set params because buffer has the wrong size."};
    }

    CUDA_CHECK_THROW(cudaMemcpy(m_params_inference, params, sizeof(PARAMS_T) * n_params, device_ptr ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));
    CUDA_CHECK_THROW(cudaMemcpy(m_params, m_params_inference, sizeof(PARAMS_T) * n_params, cudaMemcpyDeviceToDevice));

    parallel_for_gpu(n_params, [params_fp = m_params_full_precision, params_inference = m_params_inference] __device__(size_t i) {
      params_fp[i] = (float)params_inference[i];
    });

    CUDA_CHECK_THROW(cudaDeviceSynchronize());
  }

  std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> model() {
    return m_model;
  }

  json serialize(bool serialize_optimizer = false) {
    size_t n_params = m_model->n_params();

    json data;
    data["n_params"] = n_params;
    data["params_type"] = type_to_string<PARAMS_T>();
    data["params_binary"] = gpu_memory_to_json_binary(m_params_inference, sizeof(PARAMS_T) * n_params);

    if (serialize_optimizer) {
      data["optimizer"] = m_optimizer->serialize();
    }

    return data;
  }

  void deserialize(const json& data) {
    std::string type = data.value("params_type", type_to_string<PARAMS_T>());
    if (type == "float") {
      GPUMemory<float> params = data["params_binary"];
      set_params_full_precision(params.data(), params.size(), true);
    } else if (type == "__half") {
      GPUMemory<__half> params_hp = data["params_binary"];
      size_t n_params = params_hp.size();

      GPUMemory<PARAMS_T> params(n_params);
      parallel_for_gpu(n_params, [params = params.data(), params_hp = params_hp.data()] __device__(size_t i) {
        params[i] = (PARAMS_T)params_hp[i];
      });

      set_params(params.data(), params.size(), true);
    } else {
      throw std::runtime_error{"Trainer: snapshot parameters must be of type float of __half"};
    }

    if (data.contains("optimizer")) {
      m_optimizer->deserialize(data["optimizer"]);
    }

    reset_param_pointers();
    CUDA_CHECK_THROW(cudaDeviceSynchronize());
  }

  void set_param_gradients_pointer(PARAMS_T* gradients) {
    reset_param_pointers();
    m_model->set_params(m_params, m_params_inference, gradients);
  }

  void reset_param_pointers() {
    size_t n_params = m_model->n_params();

    m_params_full_precision = (float*)(m_params_buffer.data());
    m_params = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params);
    m_param_gradients = (PARAMS_T*)(m_params_buffer.data() + sizeof(float) * n_params + sizeof(PARAMS_T) * n_params);

    // Use the optimizer's custom params for inference, if they exist.
    m_params_inference = m_optimizer ? m_optimizer->custom_weights() : nullptr;
    if (m_params_inference == nullptr) {
      m_params_inference = m_params;
    }

    m_model->set_params(m_params, m_params_inference, m_param_gradients);
  }

  size_t n_params() const {
    return m_model->n_params();
  }

 private:
  std::shared_ptr<DifferentiableObject<T, PARAMS_T, COMPUTE_T>> m_model;
  std::shared_ptr<Optimizer<PARAMS_T>> m_optimizer;
  std::shared_ptr<Loss<COMPUTE_T>> m_loss;

  CudaGraph m_graph;

  GPUMemory<char> m_params_buffer;

  float* m_params_full_precision = nullptr;
  PARAMS_T* m_params_inference = nullptr;
  PARAMS_T* m_params = nullptr;
  PARAMS_T* m_param_gradients = nullptr;

  float m_perturbation_sigma;

  std::shared_ptr<ForwardContext> m_training_ctx;

  pcg32 m_rng;
};

}  // namespace tcnn
