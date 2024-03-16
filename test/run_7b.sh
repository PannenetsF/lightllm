#! /bin/bash 

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/workspace/lightllm-renew
export C=2000
export host=0.0.0.0
export port=12445
export ncclport=21456

export HTTP_PROXY=
export http_proxy=
export HTTPS_PROXY=
export https_proxy=

for pp in 10 20 50 100 200 2000 
do 
    for chunk_size in 64 128 256 
    do
        targetname="7b_${C}_${pp}_bib_chunk_$chunk_size.json"
        if [ -f $targetname ]; then
            echo "File $targetname exists."
            # continue
            rm $targetname
        fi
        python test/kill_by_gpu.py ${CUDA_VISIBLE_DEVICES}
        python -m lightllm.server.api_server --model_dir /data/models/llama2-7b-chat/ --host 0.0.0.0 --port ${port} --nccl_port ${ncclport} --tp 1 --max_total_token_num 120000 --max_req_total_len 4096 --mode bib_decoding_cuda,bib_decoding_cuda=$chunk_size --bib_route --bib_size 500 --tokenizer_mode fast &
        while true; do
            # 使用curl尝试连接到指定的主机和端口
            response=$(curl -s -o /dev/null -w "%{http_code}" "$host:$port")
            echo $response
            if curl -s -o /dev/null "$host:$port"; then
                echo "服务在端口 $port 上运行"
                break  # 退出循环，因为服务可用
            else
                echo "等待服务在端口 $port 上运行..."
                sleep 2  # 等待5秒后再次尝试连接
            fi
        done
        python test/benchmark_n_clients_ad.py --addr 0.0.0.0 --port ${port} --dataset /data/fanyunqian/ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer /data/models/llama2-7b-chat/ --num-clients ${pp} --num-prompts ${C} --mode unknown_output_len --save_name $targetname
        python test/kill_by_gpu.py ${CUDA_VISIBLE_DEVICES}
        ps aux | grep api_server || awk '{print $2}' | xargs kill -9
        sleep 10
        echo "Done with chunk size $chunk_size"
        echo "----------------------------------"
        port=$((port+10))
        ncclport=$((ncclport+10))
    done


    targetname="7b_${C}_${pp}_light_bib.json"
    if [ -f $targetname ]; then
        echo "File $targetname exists."
        # continue
        rm $targetname
    fi
    python test/kill_by_gpu.py ${CUDA_VISIBLE_DEVICES}
    python -m lightllm.server.api_server --model_dir /data/models/llama2-7b-chat/ --host 0.0.0.0 --port ${port} --mode bib_decoding_cuda,bib_decoding_cuda=256 --tp 1 --max_total_token_num 120000 --max_req_total_len 4096 --tokenizer_mode fast &
    while true; do
        # 使用curl尝试连接到指定的主机和端口
        response=$(curl -s -o /dev/null -w "%{http_code}" "$host:$port")
        echo $response
        if curl -s -o /dev/null "$host:$port"; then
            echo "服务在端口 $port 上运行"
            break  # 退出循环，因为服务可用
        else
            echo "等待服务在端口 $port 上运行..."
            sleep 2  # 等待5秒后再次尝试连接
        fi
    done
    python test/benchmark_n_clients_ad.py --addr 0.0.0.0 --port ${port} --dataset /data/fanyunqian/ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer /data/models/llama2-7b-chat/ --num-clients ${pp} --num-prompts ${C} --mode unknown_output_len --save_name $targetname
    python test/kill_by_gpu.py ${CUDA_VISIBLE_DEVICES}
    ps aux | grep api_server || awk '{print $2}' | xargs kill -9
    sleep 10
    echo "Done with lightllm + bib"
    echo "----------------------------------"
    port=$((port+10))
    ncclport=$((ncclport+10))


    targetname="7b_${C}_${pp}_light_pure.json"
    if [ -f $targetname ]; then
        echo "File $targetname exists."
        # continue
        rm $targetname
    fi
    python test/kill_by_gpu.py ${CUDA_VISIBLE_DEVICES}
    python -m lightllm.server.api_server --model_dir /data/models/llama2-7b-chat/ --host 0.0.0.0 --port ${port} --tp 1 --max_total_token_num 120000 --max_req_total_len 4096 --tokenizer_mode fast &
    while true; do
        # 使用curl尝试连接到指定的主机和端口
        response=$(curl -s -o /dev/null -w "%{http_code}" "$host:$port")
        echo $response
        if curl -s -o /dev/null "$host:$port"; then
            echo "服务在端口 $port 上运行"
            break  # 退出循环，因为服务可用
        else
            echo "等待服务在端口 $port 上运行..."
            sleep 2  # 等待5秒后再次尝试连接
        fi
    done
    python test/benchmark_n_clients_ad.py --addr 0.0.0.0 --port ${port} --dataset /data/fanyunqian/ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer /data/models/llama2-7b-chat/ --num-clients ${pp} --num-prompts ${C} --mode unknown_output_len --save_name $targetname
    python test/kill_by_gpu.py ${CUDA_VISIBLE_DEVICES}
    ps aux | grep api_server || awk '{print $2}' | xargs kill -9
    sleep 10
    echo "Done with lightllm + bib"
    echo "----------------------------------"
    port=$((port+10))
    ncclport=$((ncclport+10))
done 
