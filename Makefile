target_colab_rete:
	nvcc -arch=$(CUDA_COMPUTE_CAPABILITY) src/Graph.cu src/utils.cu src/WeightedGraph.cu src/rete_cpu.cu src/rete_gpu.cu test/main_rete.cu -o rete_main 
