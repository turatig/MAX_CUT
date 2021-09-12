target_colab_main:
	nvcc -arch=$(CUDA_COMPUTE_CAPABILITY) src/lorena_batch.cu src/Graph.cu src/utils.cu  src/rete_cpu.cu src/rete_gpu.cu src/lorena_cpu.cu src/lorena_gpu.cu test/main.cu -o main 

target_colab_main_parallel:
	nvcc -arch=$(CUDA_COMPUTE_CAPABILITY) src/Graph.cu src/utils.cu src/rete_gpu.cu src/lorena_cpu.cu src/lorena_gpu.cu src/lorena_batch.cu test/main_parallel.cu -o main_parallel 
