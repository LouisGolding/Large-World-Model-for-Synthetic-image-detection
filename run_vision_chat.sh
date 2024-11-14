#! /bin/bash

export SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd $PROJECT_DIR
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR"

export llama_tokenizer_path="llama_checkpoint"
export vqgan_checkpoint="vqgan_flax_checkpoint.pkl"
export lwm_checkpoint="lwm_checkpoint/pytorch_model-00001-of-00002.bin"
export input_file="sets/train_dataset.json"

# Relevant params
# --input_file: A given image file (png or jpg) or video file (any video format support by decord, e.g. mp4)
# --max_n_frames: Maximum number of frames to process. If the video is longer than max_n_frames frames, it uniformly samples max_n_frames frames from the video

python3 -u -m lwm.vision_chat \
    --prompt="Is this image real or fake?" \
    --input_file="$input_file" \
    --vqgan_checkpoint="$vqgan_checkpoint" \
    --mesh_dim='!1,1,-1,1' \
    --dtype='fp32' \
    --load_llama_config='7b' \
    --max_n_frames=8 \
    --update_llama_config="dict(sample_mode='text',theta=50000000,max_sequence_length=131072,scan_attention=False,scan_query_chunk_size=128,scan_key_chunk_size=128,scan_mlp=False,scan_mlp_chunk_size=2048,scan_layers=True)" \
    --load_checkpoint="params::$lwm_checkpoint" \
    --tokenizer="$llama_tokenizer_path" \
2>&1 | tee ~/output.log
read


