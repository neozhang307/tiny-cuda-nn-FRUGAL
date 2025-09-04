# CLAUDE.md - Claude Code Guidelines for tiny-cuda-nn FRUGAL Integration

## Implementation Guidelines

**IMPORTANT: Don't implement anything until the user understands the implementation plan and confirms the plan**

- Always present implementation plans first
- Wait for user confirmation before proceeding with code changes
- Explain the approach and rationale before taking action
- Get approval for any file creation or modification

## Project Context
- This is a tiny-cuda-nn project with FRUGAL memory optimization integration
- FRUGAL is included as a git submodule for GPU memory usage optimization
- The main application learns and generates images using neural networks
- Integration uses an adapter pattern in `include/memopt-adapter/adapter.h`

## Build Environment
- Requires FRUGAL conda environment: `source ~/miniconda3/bin/activate && conda activate frugal`
- Build commands: `make clean && make config && make build`
- Main executable: `./build/mlp_learning_an_image`

## Current Status (Sept 4, 2025)
- Basic integration is working and functional
- Achieves **81% memory reduction** (32GB → 6GB) with **35% performance overhead**
- Performance matches FRUGAL predictions: 0.81s per iteration (81s for 100 iterations)
- Validation completed: Image comparison script confirms correctness

### Known Issues
- Memory ownership conflict causes cleanup error after execution (doesn't affect computation results)
- `cudaDeviceSynchronize()` inside execution loop prevents GPU pipelining
- Inference fails due to memory corruption from FRUGAL's memory management

### Integration Architecture
- **Adapter Pattern**: `include/memopt-adapter/adapter.h` bridges tiny-cuda-nn and FRUGAL
- **Memory Registration**: All GPU matrices registered with FRUGAL's MemoryManager
- **Task Execution**: Training operations wrapped as FRUGAL tasks for optimization
- **Current Pattern**: Single-shot "optimize & shrink" for entire training run

### Performance Analysis
- **Model Size**: ~118M parameters (4096 neurons × 8 layers)
- **Batch Size Impact**: 262,144 samples/batch drives 32GB memory usage (not model weights)
- **Optimization**: FRUGAL successfully reduces peak memory by moving data between iterations
- **Bottleneck**: Per-iteration synchronization, not the optimization itself

## Future Development (Postponed)
### Planned Improvements
1. **Iterative Pattern**: Implement "optimize & enlarge" for repeated execution cycles
2. **Repeated Graph Support**: Ensure same arrays resident at begin/end of iterations
3. **New Architecture**: Migrate to executor_v2 and new MemoryManager
4. **Memory Fix**: Resolve dual ownership between FRUGAL and tiny-cuda-nn
5. **Launch Graph Reimplementation**: Eventually need to reimplement `executeGraphRepeatedly` for iterative optimization
   - Current implementation launches full graph repeatedly (100 iterations)
   - For demonstration purposes, could be simplified to 1 iteration to focus on iterative optimization mechanics
   - Would enable testing optimize/shrink/enlarge cycle without long execution times

### Why Postponed
- Current implementation sufficient for research validation
- Successfully demonstrates FRUGAL can optimize real ML workloads
- Core functionality proven despite cleanup issues
- Other urgent tasks take priority