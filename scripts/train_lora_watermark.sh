# model="runwayml/stable-diffusion-v1-5"
model="/home/mlsnrs/data/common_model/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"

accelerate launch --gpu_ids='6'  train_network.py \
    --pretrained_model_name_or_path=$model \
    --dataset_config="/home/mlsnrs/data/cpy/sd-scripts/config/pokemon_watermark.toml" \
    --output_dir="/home/mlsnrs/data/cpy/sd-scripts/results/pokemon/watermark" \
    --output_name="pokemon_watermark" \
    --save_model_as=safetensors \
    --prior_loss_weight=1.0 \
    --max_train_epochs=80 \
    --learning_rate=1e-4 \
    --optimizer_type="AdamW" \
    --mixed_precision="fp16" \
    --save_every_n_epochs=20 \
    --network_module=networks.lora \
    --network_dim=64 \
    --gradient_checkpointing \
    --gradient_accumulation_steps=1 \
    --cache_latents

# TODO: 原文写的是微调30个epoch

# network_dim => lora rank
