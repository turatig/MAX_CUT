target_colab:
	nvcc -arch=$(CUDA_COMPUTE_CAPABILITY) src/Graph.cu src/utils.cu  src/rete_cpu.cu src/rete_gpu.cu src/lorena_cpu.cu src/lorena_gpu.cu test/main.cu -o main 
