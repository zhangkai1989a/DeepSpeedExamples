export LOCAL_RANK=0
DATASET_OPTS="--dataset-name random --random-input-len 512 --random-output-len 32 --random-range-ratio 1.0"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
BATCH_SIZE=1

# python -m sglang.bench_offline_throughput --model-path ${MODEL_NAME}  ${DATASET_OPTS} --num-prompts ${BATCH_SIZE} --disable-cuda-graph

MODEL_NAME="meta-llama/Llama-3.2-1B"
python -m sglang.bench_offline_throughput --model-path ${MODEL_NAME}  ${DATASET_OPTS} --num-prompts ${BATCH_SIZE} --disable-cuda-graph --zero-inference-config ds_offload_cpu.json
python -m sglang.bench_offline_throughput --model-path ${MODEL_NAME}  ${DATASET_OPTS} --num-prompts ${BATCH_SIZE} --disable-cuda-graph --zero-inference-config ds_offload_nvme_aio.json
python -m sglang.bench_offline_throughput --model-path ${MODEL_NAME}  ${DATASET_OPTS} --num-prompts ${BATCH_SIZE} --disable-cuda-graph --zero-inference-config ds_offload_nvme_gds.json
python -m sglang.bench_offline_throughput --model-path ${MODEL_NAME}  ${DATASET_OPTS} --num-prompts ${BATCH_SIZE} --disable-cuda-graph


MODEL_NAME="meta-llama/Meta-Llama-3.1-70B"
# python -m sglang.bench_offline_throughput --model-path ${MODEL_NAME}  ${DATASET_OPTS} --num-prompts ${BATCH_SIZE} --disable-cuda-graph --zero-inference
# python -m sglang.bench_offline_throughput --model-path ${MODEL_NAME}  ${DATASET_OPTS} --num-prompts ${BATCH_SIZE} --disable-cuda-graph
