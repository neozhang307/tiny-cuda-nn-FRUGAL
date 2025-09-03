#pragma once

#include <tiny-cuda-nn/gpu_matrix.h>

#include <functional>
#include <memopt.hpp>

namespace memopt_adapter {

inline std::vector<tcnn::GPUMatrixBase *> managedMatrices;

template <typename T>
void register_array(tcnn::GPUMatrixDynamic<T> &matrix, bool input = false, bool output = false) {
  managedMatrices.push_back(&matrix);

  memopt::MemoryManager::getInstance().registerManagedMemoryAddress(matrix.data(), matrix.n_bytes());
  if (input) {
    memopt::MemoryManager::getInstance().registerApplicationInput(matrix.data());
  }
  if (output) {
    memopt::MemoryManager::getInstance().registerApplicationOutput(matrix.data());
  }
}

inline size_t total_registered_array_bytes() {
  size_t bytes = 0;
  const auto& memoryInfos = memopt::MemoryManager::getInstance().getMemoryArrayInfos();
  for (const auto& info : memoryInfos) {
    bytes += info.size;
  }
  return bytes;
}

typedef std::function<void(cudaStream_t)> Task;

inline std::vector<Task> tasks;

inline void register_and_execute_task(
  std::vector<void *> inputs,
  std::vector<void *> outputs,
  Task task,
  cudaStream_t stream
) {
  auto taskId = tasks.size();
  tasks.push_back(task);
  memopt::annotateNextTask(taskId, inputs, outputs, stream);
  task(stream);
}

inline void execute_random_task(int taskId, std::map<void *, void *> addressUpdateMap, cudaStream_t stream) {
  for (auto matrix : managedMatrices) {
    matrix->try_updating_address(addressUpdateMap);
  }
  tasks[taskId](stream);
}

}  // namespace memopt_adapter
