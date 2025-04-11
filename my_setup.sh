# Change cosmos-predict1.yaml content to:

# name: cosmos-predict1
# channels:
#   - conda-forge
#   - nvidia
#   - defaults
# dependencies:
#   - python=3.10
#   - pip
#   - cmake
#   - ninja
#   - pip:
#       - torch
#       - torchvision
#       - torchaudio

module load cudatoolkit/12.4.0
conda env create --file cosmos-predict1.yaml
module load anaconda
conda activate cosmos-predict1
pip install -r requirements.txt

# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10

# Install Transformer engine.
pip install transformer-engine[pytorch]==1.12.0

# Install Apex for full training with bfloat16.
# git clone https://github.com/NVIDIA/apex
# cd apex
# CUDA_HOME=$CONDA_PREFIX pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" .

# Download model weights
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_checkpoints.py --model_sizes 7B --model_types Video2World --checkpoint_dir /network/scratch/a/anthony.gosselin/Models

# Single GPU inference
module load anaconda
conda activate cosmos-predict1
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world.py \
    --checkpoint_dir /network/scratch/a/anthony.gosselin/Models \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World \
    --input_image_or_video_path assets/diffusion/video2world_input0.jpg \
    --num_input_frames 1 \
    --offload_prompt_upsampler \
    --disable_guardrail \
    --video_save_name diffusion-video2world-7b


# Multi-GPU inference
export NUM_GPUS_INF=4
export PROMPT="The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features dry, grassy fields and rolling hills under a partly cloudy sky. The scene conveys a sense of open space and tranquility."
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPUS_INF cosmos_predict1/diffusion/inference/video2world.py \
    --num_gpus $NUM_GPUS_INF \
    --checkpoint_dir /network/scratch/a/anthony.gosselin/Models \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World \
    --input_image_or_video_path assets/diffusion/video2world_input0.jpg \
    --num_input_frames 1 \
    --offload_prompt_upsampler \
    --offload_guardrail_models \
    --disable_guardrail \
    --disable_prompt_upsampler \
    --prompt "$PROMPT" \
    --video_save_name diffusion-video2world-7b

###--- Other tests

module load anaconda
conda activate cosmos-predict1
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world.py \
    --checkpoint_dir /network/scratch/a/anthony.gosselin/Models \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World \
    --input_image_or_video_path /home/mila/a/anthony.gosselin/dev/Cosmos/samples/48_48_90021.mp4 \
    --prompt "The setting is a multi-lane arterial road situated in a mountainous area. The sky is overcast, suggesting cloudy weather conditions. The road is bordered by lush greenery and trees on both sides, with a guardrail visible on the left side. Several vehicles are present, including a white sedan in the center lane and a silver sedan in the right lane. A large white truck is also traveling in the same direction as the sedans. The road appears smooth, and there are no visible pedestrians or other obstructions. An accident occurred involving a truck and a car. The lead vehicle, a truck, decelerated unexpectedly. Due to the driver being distracted, the following car failed to maintain a safe distance and hit the truck from behind." \
    --num_input_frames 9 \
    --offload_prompt_upsampler \
    --disable_prompt_upsampler \
    --disable_guardrail \
    --video_save_name diffusion-video2world-7b

    # Same thing but no crash
    CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world.py \
    --checkpoint_dir /network/scratch/a/anthony.gosselin/Models \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World \
    --input_image_or_video_path /home/mila/a/anthony.gosselin/dev/Cosmos/samples/48_48_90021.mp4 \
    --prompt "The setting is a multi-lane arterial road situated in a mountainous area. The sky is overcast, suggesting cloudy weather conditions. The road is bordered by lush greenery and trees on both sides, with a guardrail visible on the left side. Several vehicles are present, including a white sedan in the center lane and a silver sedan in the right lane. A large white truck is also traveling in the same direction as the sedans. The road appears smooth, and there are no visible pedestrians or other obstructions. A car nearly hits the rear-end of a truck. The lead vehicle, a truck, decelerated unexpectedly. Due to the driver being distracted, the following car failed to maintain a safe distance and almost hit the truck from behind. Luckily, it managed to brake very suddenfly and avoid an accident." \
    --num_input_frames 9 \
    --offload_prompt_upsampler \
    --disable_prompt_upsampler \
    --disable_guardrail \
    --video_save_name diffusion-video2world-7b

    # 11_199
    CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/video2world.py \
    --checkpoint_dir /network/scratch/a/anthony.gosselin/Models \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World \
    --input_image_or_video_path /home/mila/a/anthony.gosselin/dev/Cosmos/samples/11_11_90199.mp4 \
    --prompt "The ego vehicle was involved as it was hit by the reversing car. This occurred on a mountain road with a curve, where the reversing vehicle did not properly account for the presence of other vehicles, leading to the impact" \
    --num_input_frames 9 \
    --offload_prompt_upsampler \
    --disable_prompt_upsampler \
    --disable_guardrail \
    --video_save_name diffusion-video2world-7b

export NUM_GPUS_INF=4
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPUS_INF cosmos_predict1/diffusion/inference/video2world.py \
    --num_gpus $NUM_GPUS_INF \
    --checkpoint_dir /network/scratch/a/anthony.gosselin/Models \
    --diffusion_transformer_dir Cosmos-Predict1-7B-Video2World \
    --input_image_or_video_path /home/mila/a/anthony.gosselin/dev/Cosmos/samples/48_48_90021.mp4 \
    --prompt "The setting is a multi-lane arterial road situated in a mountainous area. The sky is overcast, suggesting cloudy weather conditions. The road is bordered by lush greenery and trees on both sides, with a guardrail visible on the left side. Several vehicles are present, including a white sedan in the center lane and a silver sedan in the right lane. A large white truck is also traveling in the same direction as the sedans. The road appears smooth, and there are no visible pedestrians or other obstructions. An accident occurred involving a truck and a car. The lead vehicle, a car, decelerated unexpectedly. Due to the driver being distracted, the following truck failed to maintain a safe distance and hit the car from behind." \
    --disable_prompt_upsampler \
    --num_input_frames 9 \
    --offload_prompt_upsampler \
    --disable_guardrail \
    --video_save_name diffusion-video2world-7b

# AR model inference

export HF_HOME="/network/scratch/a/anthony.gosselin/Models"

module load cudatoolkit/12.4.0
module load anaconda
conda activate cosmos-predict1
cd /home/mila/a/anthony.gosselin/dev/cosmos-predict1

# On post-trained model
PROMPT="The setting is a highway at a T-junction under sunny conditions with clear visibility. A beige car is positioned prominently in the foreground, angled slightly as if in motion. The car has a roof rack and appears to be traveling along the main road. In the background, there are trees lining the roadside, indicating a rural or semi-rural environment. The road markings include arrows guiding traffic flow, suggesting the presence of a controlled intersection. No other vehicles or pedestrians are immediately visible, and the sky is clear. At a non-signalized T-junction on a highway, the ego vehicle proceeded straight while crossing paths with another car. The ego vehicle and the car failed to notice each other due to a lack of awareness, resulting in their paths intersecting dangerously and colliding."
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/autoregressive/inference/video2world.py \
    --checkpoint_dir /network/scratch/a/anthony.gosselin/Models \
    --ar_model_dir 'Cosmos-AR-5B-Video2World--epoch=13-step=89999-last' \
    --input_image_or_video_path /home/mila/a/anthony.gosselin/dev/Cosmos/samples/10_04185_9frames.mp4 \
    --prompt "${PROMPT}" \
    --num_input_frames 9 \
    --disable_guardrail \
    --video_save_name AR-video2world-7b

# 43_02184
# "The scene depicts an urban intersection under clear weather conditions. A multi-story building labeled stands prominently in the background. Several other buildings, including a red brick structure and a beige high-rise, surround the area. Overhead power lines crisscross above the street. A white bus marked with the number 9 is present along with several cars, including a white sedan and a black vehicle. A pedestrian crossing is visible, and a cyclist is riding near the intersection. Traffic signals and road signs are also noticeable, indicating a regulated traffic environment. A vehicle turns left across the path of a bus at a signalized junction. Both vehicles were traveling too fast, resulting in a short braking distance and causing the vehicles to collide. The ego vehicle is not involved in the accident."

# Multi-gpu inference
NUM_GPUS=4
PROMPT="The setting is a highway at a T-junction under sunny conditions with clear visibility. A beige car is positioned prominently in the foreground, angled slightly as if in motion. The car has a roof rack and appears to be traveling along the main road. In the background, there are trees lining the roadside, indicating a rural or semi-rural environment. The road markings include arrows guiding traffic flow, suggesting the presence of a controlled intersection. No other vehicles or pedestrians are immediately visible, and the sky is clear. At a non-signalized T-junction on a highway, the ego vehicle proceeded straight while crossing paths with another car. The ego vehicle and the car failed to notice each other due to a lack of awareness, resulting in their paths intersecting dangerously and colliding."
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) torchrun --nproc_per_node=${NUM_GPUS} cosmos_predict1/autoregressive/inference/video2world.py \
    --num_gpus ${NUM_GPUS} \
    --checkpoint_dir /network/scratch/a/anthony.gosselin/Models \
    --ar_model_dir 'Cosmos-AR-5B-Video2World--epoch=13-step=89999-last' \
    --input_type text_and_video \
    --input_image_or_video_path /home/mila/a/anthony.gosselin/dev/Cosmos/samples/10_04185_9frames.mp4 \
    --prompt "${PROMPT}" \
    --num_input_frames 9 \
    --top_p 0.7 \
    --temperature 1.0 \
    --disable_guardrail \
    --video_save_name AR-video2world-7b-trained


# POST-Train Diffusion model
export OUTPUT_ROOT="/network/scratch/a/anthony.gosselin/Models" # default value
torchrun --nproc_per_node=4 -m cosmos_predict1.diffusion.training.train \
    --config=cosmos_predict1/diffusion/training/config/config.py -- experiment=video2world_7b_example_hdvila