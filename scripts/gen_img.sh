# lora gen

python gen_img_diffusers.py \
    --ckpt "/home/mlsnrs/data/common_model/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14" \
    --network_module networks.lora \
    --network_weights "./results/pokemon/watermark/pokemon_watermark.safetensors" \
    --network_mul 0.8 \
    --outdir "./results/pokemon/watermark/img" \
    --scale 8 \
    --steps 48 \
    --xformers \
    --W 512 \
    --H 512 \
    --fp16 \
    --sampler k_euler_a \
    --clip_skip 2 \
    --max_embeddings_multiples 1 \
    --batch_size 8 \
    --images_per_prompt 1 \
    --interactive
