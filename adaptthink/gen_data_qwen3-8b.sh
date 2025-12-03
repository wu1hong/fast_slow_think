# Initialize VLLM server. You can start multiple servers to accelerate pre-sampling.
# Start VLLM server in the background
# CUDA_VISIBLE_DEVICES=0 nohup vllm serve Qwen/Qwen3-8B --served_model_name Qwen3-8B-1 --port 8001 > /dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup vllm serve Qwen/Qwen3-8B --served_model_name Qwen3-8B-2 --port 8002 > /dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup vllm serve Qwen/Qwen3-8B --served_model_name Qwen3-8B-3 --port 8003 > /dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup vllm serve Qwen/Qwen3-8B --served_model_name Qwen3-8B-4 --port 8004 > /dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup vllm serve Qwen/Qwen3-8B --served_model_name Qwen3-8B-5 --port 8005 > /dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup vllm serve Qwen/Qwen3-8B --served_model_name Qwen3-8B-6 --port 8006 > /dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup vllm serve Qwen/Qwen3-8B --served_model_name Qwen3-8B-7 --port 8007 > /dev/null 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup vllm serve Qwen/Qwen3-8B --served_model_name Qwen3-8B-8 --port 8008 > /dev/null 2>&1 &

# Sampling 16 responses for each training problem.
mkdir -p log
python presampling_ref_responses.py --K 16 --dataset_path ../data/data1202.jsonl --model_name Qwen3-8B-1 --max_tokens 16384 --start_id 0     --end_id 2500  --port 8001 > ./log/Qwen3-8B-1.log 2>&1 &
python presampling_ref_responses.py --K 16 --dataset_path ../data/data1202.jsonl --model_name Qwen3-8B-2 --max_tokens 16384 --start_id 2500  --end_id 5000  --port 8002 > ./log/Qwen3-8B-2.log 2>&1 &
python presampling_ref_responses.py --K 16 --dataset_path ../data/data1202.jsonl --model_name Qwen3-8B-3 --max_tokens 16384 --start_id 5000  --end_id 7500  --port 8003 > ./log/Qwen3-8B-3.log 2>&1 &
python presampling_ref_responses.py --K 16 --dataset_path ../data/data1202.jsonl --model_name Qwen3-8B-4 --max_tokens 16384 --start_id 7500  --end_id 10000 --port 8004 > ./log/Qwen3-8B-4.log 2>&1 &
python presampling_ref_responses.py --K 16 --dataset_path ../data/data1202.jsonl --model_name Qwen3-8B-5 --max_tokens 16384 --start_id 10000 --end_id 12500 --port 8005 > ./log/Qwen3-8B-5.log 2>&1 &
python presampling_ref_responses.py --K 16 --dataset_path ../data/data1202.jsonl --model_name Qwen3-8B-6 --max_tokens 16384 --start_id 12500 --end_id 15000 --port 8006 > ./log/Qwen3-8B-6.log 2>&1 &
python presampling_ref_responses.py --K 16 --dataset_path ../data/data1202.jsonl --model_name Qwen3-8B-7 --max_tokens 16384 --start_id 15000 --end_id 17500 --port 8007 > ./log/Qwen3-8B-7.log 2>&1 &
python presampling_ref_responses.py --K 16 --dataset_path ../data/data1202.jsonl --model_name Qwen3-8B-8 --max_tokens 16384 --start_id 17500 --end_id 22667 --port 8008 > ./log/Qwen3-8B-8.log 2>&1 &

# Postprocess to get instance-level accuracy
# python src/postprocess_ref_results.py --input_path ./data/train/ref_presampling/DeepSeek-R1-Distill-Qwen-1.5B_deepscaler_n0_K16_len16384.json --output_path ./data/train/ref_results/DeepSeek-R1-Distill-Qwen-1.5B_deepscaler_K16_len16384.json