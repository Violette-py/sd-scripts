本脚本是基于Diffusers的推理（图像生成）脚本，支持SD 1.x和2.x模型、本仓库训练的LoRA、ControlNet（仅确认v1.0可用）等。通过命令行使用。

# 概要

* 基于Diffusers (v0.10.2) 的推理（图像生成）脚本。
* 支持SD 1.x和2.x (base/v-parameterization)模型。
* 支持txt2img、img2img、inpainting。
* 支持交互模式、从文件读取prompt、连续生成。
* 可指定每行prompt生成的图片数量。
* 可指定整体的重复次数。
* 支持`fp16`和`bf16`。
* 支持xformers，可实现高速生成。
    * 通过xformers可节省显存，但没有Automatic 1111的Web UI优化得好，生成512*512图片大约需要6GB显存。
* 支持将prompt扩展到225 token。支持negative prompt和权重。
* 支持Diffusers的多种sampler（比Web UI少一些）。
* 支持Text Encoder的clip skip（使用倒数第n层的输出）。
* 支持单独加载VAE。
* 支持CLIP Guided Stable Diffusion、VGG16 Guided Stable Diffusion、Highres. fix、upscale。
    * Highres. fix为独立实现，未参考Web UI，输出结果可能不同。
* 支持LoRA。可指定应用率、同时使用多个LoRA、权重合并。
    * 不能分别为Text Encoder和U-Net指定不同应用率。
* 支持Attention Couple。
* 支持ControlNet v1.0。
* 不能在中途切换模型，但可通过批处理脚本实现。
* 添加了许多个人需要的功能。

由于功能增加时未做全部测试，可能会影响旧功能。如有问题请反馈。

# 基本用法

## 交互模式生成图片

请按如下方式输入：

```batchfile
python gen_img_diffusers.py --ckpt <模型名> --outdir <图片输出目录> --xformers --fp16 --interactive
```

`--ckpt`指定模型（Stable Diffusion的checkpoint文件、Diffusers模型文件夹或Hugging Face模型ID），`--outdir`指定图片输出目录。

`--xformers`指定使用xformers（不使用可去掉）。`--fp16`指定使用fp16推理。RTX 30系GPU可用`--bf16`指定bf16推理。

`--interactive`指定交互模式。

使用Stable Diffusion 2.0（或其衍生模型）时需加`--v2`。使用v-parameterization模型（如`768-v-ema.ckpt`及其衍生模型）时还需加`--v_parameterization`。

`--v2`指定错误会导致加载模型时报错。`--v_parameterization`指定错误会生成棕色图片。

出现`Type prompt:`时输入prompt。

![image](https://user-images.githubusercontent.com/52813779/235343115-f3b8ac82-456d-4aab-9724-0cc73c4534aa.png)

※若图片无法显示且报错，可能安装了无界面的OpenCV，请用`pip install opencv-python`安装普通OpenCV，或加`--no_preview`关闭图片显示。

选中图片窗口后按任意键关闭窗口，可继续输入prompt。输入prompt时按Ctrl+Z再回车可退出脚本。

## 单个prompt批量生成图片

如下输入（实际为一行）：

```batchfile
python gen_img_diffusers.py --ckpt <模型名> --outdir <图片输出目录> 
    --xformers --fp16 --images_per_prompt <生成数量> --prompt "<prompt>"
```

`--images_per_prompt`指定每个prompt生成的图片数。`--prompt`指定prompt，含空格时需用双引号。

可用`--batch_size`指定batch size（见后文）。

## 从文件读取prompt批量生成

如下输入：

```batchfile
python gen_img_diffusers.py --ckpt <模型名> --outdir <图片输出目录> 
    --xformers --fp16 --from_file <prompt文件名>
```

`--from_file`指定包含prompt的文件，每行一个prompt。可用`--images_per_prompt`指定每行生成数量。

## 使用negative prompt和权重

在prompt中用`--x`（如`--n`）指定后，后面为negative prompt。

支持与AUTOMATIC1111 Web UI相同的 `()`、`[]`、`(xxx:1.3)` 等权重写法（实现参考Diffusers的[Long Prompt Weighting Stable Diffusion](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#long-prompt-weighting-stable-diffusion)）。

无论命令行、文件读取都可用。

![image](https://user-images.githubusercontent.com/52813779/235343128-e79cd768-ec59-46f5-8395-fce9bdc46208.png)

# 主要参数

请在命令行指定。

## 模型指定

- `--ckpt <模型名>`：指定模型名。必填。可为Stable Diffusion checkpoint、Diffusers模型文件夹或Hugging Face模型ID。

- `--v2`：使用Stable Diffusion 2.x模型时指定。1.x时不需指定。

- `--v_parameterization`：使用v-parameterization模型时指定（如`768-v-ema.ckpt`、Waifu Diffusion v1.5等）。
    
    `--v2`指定错误会报错，`--v_parameterization`指定错误会生成棕色图片。

- `--vae`：指定VAE。未指定时用模型自带VAE。

## 图片生成与输出

- `--interactive`：交互模式。输入prompt即生成图片。

- `--prompt <prompt>`：指定prompt，含空格需用双引号。

- `--from_file <prompt文件名>`：指定包含prompt的文件，每行一个prompt。图片尺寸、guidance scale等可用prompt参数指定。

- `--W <宽度>`：图片宽度，默认`512`。

- `--H <高度>`：图片高度，默认`512`。

- `--steps <步数>`：采样步数，默认`50`。

- `--scale <guidance scale>`：unconditional guidance scale，默认`7.5`。

- `--sampler <采样器名>`：指定采样器，默认`ddim`。支持ddim、pndm、dpmsolver、dpmsolver+++、lms、euler、euler_a（后3个也可用k_lms、k_euler、k_euler_a）。

- `--outdir <输出目录>`：指定图片输出目录。

- `--images_per_prompt <生成数量>`：每个prompt生成图片数，默认`1`。

- `--clip_skip <跳过层数>`：指定CLIP倒数第几层输出。默认用最后一层。

- `--max_embeddings_multiples <倍数>`：CLIP输入输出长度为默认75的几倍。未指定为75，如指定3则为225。

- `--negative_scale` : 单独指定uncoditioning guidance scale。实现参考[gcem156的文章](https://note.com/gcem156/n/ne9a53e4a6f43)。

## 显存与速度调节

- `--batch_size <batch size>`：指定batch size，默认`1`。batch size大则显存占用高但生成快。

- `--vae_batch_size <VAE batch size>`：指定VAE batch size，默认与batch size相同。VAE更占显存，若denoising后显存不足可减小此值。

- `--xformers`：指定使用xformers。

- `--fp16`：使用fp16推理。不指定fp16/bf16时为fp32。

- `--bf16`：使用bf16推理，仅RTX 30系可用。其他GPU指定会报错。bf16比fp16更不易出现NaN（全黑图片）。

## 额外网络（如LoRA）

- `--network_module`：指定额外网络。LoRA时为`--network_module networks.lora`。多个LoRA时可多次指定。

- `--network_weights`：指定额外网络权重文件。多个LoRA时用空格分隔，数量与`--network_module`一致。

- `--network_mul`：指定额外网络权重倍率，默认`1`。多个LoRA时用空格分隔，数量与`--network_module`一致。

- `--network_merge`：用`--network_mul`指定的权重提前合并额外网络。不能与`--network_pre_calc`同时用。不能用prompt参数`--am`和Regional LoRA，但生成速度与未用LoRA时相当。

- `--network_pre_calc`：每次生成前提前计算额外网络权重。可用prompt参数`--am`。生成速度与未用LoRA时相当，但计算权重需时间且显存略增。用Regional LoRA时无效。

# 主要参数示例

同一prompt批量生成64张，batch size为4：

```batchfile
python gen_img_diffusers.py --ckpt model.ckpt --outdir outputs 
    --xformers --fp16 --W 512 --H 704 --scale 12.5 --sampler k_euler_a 
    --steps 32 --batch_size 4 --images_per_prompt 64 
    --prompt "beautiful flowers --n monochrome"
```

文件中每个prompt各生成10张，batch size为4：

```batchfile
python gen_img_diffusers.py --ckpt model.ckpt --outdir outputs 
    --xformers --fp16 --W 512 --H 704 --scale 12.5 --sampler k_euler_a 
    --steps 32 --batch_size 4 --images_per_prompt 10 
    --from_file prompts.txt
```

Textual Inversion和LoRA使用示例：

```batchfile
python gen_img_diffusers.py --ckpt model.safetensors 
    --scale 8 --steps 48 --outdir txt2img --xformers 
    --W 512 --H 768 --fp16 --sampler k_euler_a 
    --textual_inversion_embeddings goodembed.safetensors negprompt.pt 
    --network_module networks.lora networks.lora 
    --network_weights model1.safetensors model2.safetensors 
    --network_mul 0.4 0.8 
    --clip_skip 2 --max_embeddings_multiples 1 
    --batch_size 8 --images_per_prompt 1 --interactive
```

# prompt参数

在prompt中可用`--n`等形式指定参数。交互、命令行、文件读取均可用。

`--n`前后需有空格。

- `--n`：指定negative prompt。

- `--w`：指定图片宽度，覆盖命令行参数。

- `--h`：指定图片高度，覆盖命令行参数。

- `--s`：指定步数，覆盖命令行参数。

- `--d`：指定seed。`--images_per_prompt`时可用逗号分隔多个seed。
    ※因多种原因，与Web UI同seed生成图片可能不同。

- `--l`：指定guidance scale，覆盖命令行参数。

- `--t`：img2img的strength，覆盖命令行参数。

- `--nl`：指定negative prompt的guidance scale，覆盖命令行参数。

- `--am`：指定额外网络权重，覆盖命令行参数。多个网络时用逗号分隔。

※指定这些参数时，batch size可能小于设定值（参数不同无法批量生成）。从文件读取prompt时，参数相同效率更高。

例：
```
(masterpiece, best quality), 1girl, in shirt and plated skirt, standing at street under cherry blossoms, upper body, [from below], kind smile, looking at another, [goodembed] --n realistic, real life, (negprompt), (lowres:1.1), (worst quality:1.2), (low quality:1.1), bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry --w 960 --h 640 --s 28 --d 1
```

![image](https://user-images.githubusercontent.com/52813779/235343446-25654172-fff4-4aaf-977a-20d262b51676.png)

# img2img

## 参数

- `--image_path`：指定img2img用图片。可为文件夹，顺序读取图片。

- `--strength`：指定img2img的strength，默认`0.8`。

- `--sequential_file_name`：生成文件名为序号，如`im_000001.png`。

- `--use_original_file_name`：生成文件名与原文件相同。

## 命令行示例

```batchfile
python gen_img_diffusers.py --ckpt trinart_characters_it4_v1_vae_merged.ckpt 
    --outdir outputs --xformers --fp16 --scale 12.5 --sampler k_euler --steps 32 
    --image_path template.png --strength 0.8 
    --prompt "1girl, cowboy shot, brown hair, pony tail, brown eyes, 
          sailor school uniform, outdoors 
          --n lowres, bad anatomy, bad hands, error, missing fingers, cropped, 
          worst quality, low quality, normal quality, jpeg artifacts, (blurry), 
          hair ornament, glasses" 
    --batch_size 8 --images_per_prompt 32
```

`--image_path`为文件夹时，顺序读取图片。生成数量为prompt数，需用`--images_per_prompt`使图片数与prompt数一致。

文件按文件名字符串排序（如`1.jpg→10.jpg→2.jpg`），建议用0补齐（如`01.jpg→02.jpg→10.jpg`）。

## img2img放大

img2img时用`--W`和`--H`指定生成图片尺寸，会先将原图resize到该尺寸再img2img。

若原图为本脚本生成，省略prompt时会自动读取元数据中的prompt，实现Highres. fix的2nd stage。

## img2img inpainting

可指定图片和mask图片进行inpainting（不支持inpainting模型，仅对mask区域img2img）。

参数如下：

- `--mask_image`：指定mask图片。可为文件夹，顺序读取。

mask为灰度图，白色区域为inpainting区域。建议边界做渐变更自然。

![image](https://user-images.githubusercontent.com/52813779/235343795-9eaa6d98-02ff-4f32-b089-80d1fc482453.png)

# 其他功能

## Textual Inversion

用`--textual_inversion_embeddings`指定embeddings（可多个）。在prompt中用去除扩展名的文件名即可调用（与Web UI一致）。negative prompt也可用。

支持本仓库训练的Textual Inversion模型和Web UI训练的模型（不支持图片embedding）。

## Extended Textual Inversion

用`--XTI_embeddings`代替`--textual_inversion_embeddings`。用法相同。

## Highres. fix

类似AUTOMATIC1111 Web UI的功能（独立实现，可能有差异）。先生成小图，再img2img生成大图，防止大分辨率下整体崩坏。

2nd stage步数为`steps*strength`。

不能与img2img同时用。

参数如下：

- `--highres_fix_scale`：启用Highres. fix，指定1st stage图片尺寸倍率。如最终1024x1024，1st stage为512x512，则`--highres_fix_scale 0.5`。与Web UI相反。

- `--highres_fix_steps`：1st stage步数，默认`28`。

- `--highres_fix_save_1st`：是否保存1st stage图片。

- `--highres_fix_latents_upscaling`：2nd stage时用latent上采样（仅支持bilinear）。未指定时用LANCZOS4。

- `--highres_fix_upscaler`：2nd stage用自定义upscaler。目前仅支持`tools.latent_upscaler`。

- `--highres_fix_upscaler_args`：传递给upscaler的参数。如`tools.latent_upscaler`时可指定权重文件。

命令行示例：

```batchfile
python gen_img_diffusers.py  --ckpt trinart_characters_it4_v1_vae_merged.ckpt
    --n_iter 1 --scale 7.5 --W 1024 --H 1024 --batch_size 1 --outdir ../txt2img 
    --steps 48 --sampler ddim --fp16 
    --xformers 
    --images_per_prompt 1  --interactive 
    --highres_fix_scale 0.5 --highres_fix_steps 28 --strength 0.5
```

## ControlNet

目前仅确认ControlNet 1.0可用。预处理仅支持Canny。

参数如下：

- `--control_net_models`：指定ControlNet模型文件。可多个，按step切换（与Web UI实现不同）。支持diff和普通模型。

- `--guide_image_path`：指定ControlNet用引导图片。可为文件夹，顺序读取。非Canny模型需提前预处理。

- `--control_net_preps`：指定ControlNet预处理。可多个。目前仅支持canny。不用预处理时指定`none`。如canny可用`--control_net_preps canny_63_191`指定两个阈值。

- `--control_net_weights`：指定ControlNet权重。可多个。

- `--control_net_ratios`：指定ControlNet应用step范围。如`0.5`为前半步应用。可多个。

命令行示例：

```batchfile
python gen_img_diffusers.py --ckpt model_ckpt --scale 8 --steps 48 --outdir txt2img --xformers 
    --W 512 --H 768 --bf16 --sampler k_euler_a 
    --control_net_models diff_control_sd15_canny.safetensors --control_net_weights 1.0 
    --guide_image_path guide.png --control_net_ratios 1.0 --interactive
```

## Attention Couple + Reginal LoRA

可将prompt分为多部分，分别指定应用于图片的区域。无单独参数，用`mask_path`和prompt指定。

先用` AND `分割prompt，最多前三部分可指定区域，后面部分应用于全图。negative prompt应用于全图。

如：

```
shs 2girls, looking at viewer, smile AND bsb 2girls, looking back AND 2girls --n bad quality, worst quality
```

准备mask图片，彩色图像，RGB各通道对应prompt的各部分。某通道全为0时，应用于全图。

如R通道为`shs 2girls, looking at viewer, smile`，G通道为`bsb 2girls, looking back`，B通道为`2girls`。如B通道未指定，则`2girls`应用于全图。

![image](https://user-images.githubusercontent.com/52813779/235343061-b4dc9392-3dae-4831-8347-1e9ae5054251.png)

mask图片用`--mask_path`指定。目前仅支持1张。会自动resize到指定尺寸。

可与ControlNet结合（推荐细致位置指定时用ControlNet）。

指定LoRA时，`--network_weights`中多个LoRA分别对应AND分割的各部分。当前要求LoRA数量与AND部分数量一致。

## CLIP Guided Stable Diffusion

基于Diffusers Community Examples的[custom pipeline](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#clip-guided-stable-diffusion)修改。

在普通prompt生成基础上，额外用更大的CLIP获取prompt文本特征，使生成图片特征更接近文本特征。需用大CLIP，显存占用高（8GB显存512*512可能不够），生成慢。

仅支持DDIM、PNDM、LMS采样器。

`--clip_guidance_scale`指定CLIP特征影响力。示例为100，可据此调整。

默认用prompt前75 token（去除权重符号）传给CLIP。用prompt参数`--c`可单独指定CLIP文本（如CLIP不识别DreamBooth identifier或"1girl"等模型特有词时，可省略）。

命令行示例：

```batchfile
python gen_img_diffusers.py  --ckpt v1-5-pruned-emaonly.ckpt --n_iter 1 
    --scale 2.5 --W 512 --H 512 --batch_size 1 --outdir ../txt2img --steps 36  
    --sampler ddim --fp16 --opt_channels_last --xformers --images_per_prompt 1  
    --interactive --clip_guidance_scale 100
```

## CLIP Image Guided Stable Diffusion

不是用文本，而是将其他图片传给CLIP，使生成图片特征更接近该图片。用`--clip_image_guidance_scale`指定影响力，`--guide_image_path`指定引导图片（文件或文件夹）。

命令行示例：

```batchfile
python gen_img_diffusers.py  --ckpt trinart_characters_it4_v1_vae_merged.ckpt
    --n_iter 1 --scale 7.5 --W 512 --H 512 --batch_size 1 --outdir ../txt2img 
    --steps 80 --sampler ddim --fp16 --opt_channels_last --xformers 
    --images_per_prompt 1  --interactive  --clip_image_guidance_scale 100 
    --guide_image_path YUKA160113420I9A4104_TP_V.jpg
```

### VGG16 Guided Stable Diffusion

使生成图片更接近指定图片。除普通prompt外，额外用VGG16获取特征，使生成图片更接近引导图片。推荐img2img使用（普通生成会偏模糊）。基于CLIP Guided Stable Diffusion机制独立实现，灵感来自VGG风格迁移。

仅支持DDIM、PNDM、LMS采样器。

`--vgg16_guidance_scale`指定VGG16特征影响力。建议从100开始调整。`--guide_image_path`指定引导图片（文件或文件夹）。

批量img2img且原图为引导图时，`--guide_image_path`和`--image_path`可相同。

命令行示例：

```batchfile
python gen_img_diffusers.py --ckpt wd-v1-3-full-pruned-half.ckpt 
    --n_iter 1 --scale 5.5 --steps 60 --outdir ../txt2img 
    --xformers --sampler ddim --fp16 --W 512 --H 704 
    --batch_size 1 --images_per_prompt 1 
    --prompt "picturesque, 1girl, solo, anime face, skirt, beautiful face 
        --n lowres, bad anatomy, bad hands, error, missing fingers, 
        cropped, worst quality, low quality, normal quality, 
        jpeg artifacts, blurry, 3d, bad face, monochrome --d 1" 
    --strength 0.8 --image_path ..\src_image
    --vgg16_guidance_scale 100 --guide_image_path ..\src_image 
```

用`--vgg16_guidance_layer`可指定VGG16用于特征提取的层（默认20，即conv4-2的ReLU）。越高层越偏风格，越低层越偏内容。

![image](https://user-images.githubusercontent.com/52813779/235343813-3c1f0d7a-4fb3-4274-98e4-b92d76b551df.png)

# 其他参数

- `--no_preview` : 交互模式下不显示预览图片。OpenCV未安装或只需查看输出文件时可用。

- `--n_iter` : 指定生成重复次数，默认1。文件读取prompt时需多次生成可用。

- `--tokenizer_cache_dir` : 指定tokenizer缓存目录。（开发中）

- `--seed` : 指定随机seed。单张时为该图片seed，多张时为生成各图片seed的随机种子（如`--from_file`批量生成，指定`--seed`可多次运行生成相同图片）。

- `--iter_same_seed` : prompt未指定seed时，`--n_iter`内每次用同一seed。用于对比不同prompt时统一seed。

- `--diffusers_xformers` : 使用Diffusers的xformers。

- `--opt_channels_last` : 推理时将tensor通道放最后，部分情况下可加速。

- `--network_show_meta` : 显示额外网络的meta信息。


--- 

# 关于Gradual Latent

Gradual Latent是一种Hires fix，会逐步增大latent的尺寸。`gen_img.py`、`sdxl_gen_img.py`、`gen_img_diffusers.py`支持如下参数：

- `--gradual_latent_timesteps`：指定开始增大latent尺寸的timestep。默认None（不使用Gradual Latent）。建议先试750。
- `--gradual_latent_ratio`：指定latent初始尺寸。默认0.5（为默认latent尺寸一半）。
- `--gradual_latent_ratio_step`：指定每次增大latent的比例。默认0.125（即0.625, 0.75, 0.875, 1.0逐步增大）。
- `--gradual_latent_ratio_every_n_steps`：指定每隔多少步增大一次latent尺寸。默认3（每3步增大一次）。

也可用prompt参数`--glt`、`--glr`、`--gls`、`--gle`指定。

__采样器必须指定`euler_a`__，因采样器源码有修改，其他采样器无效。

对SD 1.5效果更明显，SDXL效果较弱。

# Gradual Latent 说明

Gradual Latent会逐步增大latent尺寸的Hires fix。`gen_img.py`、`sdxl_gen_img.py`、`gen_img_diffusers.py`支持如下参数：

- `--gradual_latent_timesteps` : 指定开始增大latent尺寸的timestep。默认None（不使用Gradual Latent）。建议先试750。
- `--gradual_latent_ratio` : 指定latent初始尺寸。默认0.5（为默认latent尺寸一半）。
- `--gradual_latent_ratio_step`: 指定每次增大latent的比例。默认0.125（即0.625, 0.75, 0.875, 1.0逐步增大）。
- `--gradual_latent_ratio_every_n_steps`: 指定每隔多少步增大一次latent尺寸。默认3（每3步增大一次）。

也可用prompt参数`--glt`、`--glr`、`--gls`、`--gle`指定。

因采样器源码有修改，__采样器必须指定`euler_a`__，其他采样器无效。

对SD 1.5效果更明显，SDXL效果较弱。

サンプラーに手を加えているため、__サンプラーに `euler_a` を指定してください。__ 他のサンプラーでは動作しません。

SD 1.5 のほうが効果があります。SDXL ではかなり微妙です。

