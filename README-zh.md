# [ADT-Tree: Fast Inference of Visual Autoregressive Model with Adjacency-Adaptive Dynamical Draft Trees](https://arxiv.org/abs/2512.21857)

[ADT-Tree: Fast Inference of Visual Autoregressive Model](https://arxiv.org/abs/2512.21857)

<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/Lang-English-blue?style=for-the-badge" alt="English Version"></a>
  <a href="https://arxiv.org/abs/2512.21857"><img src="https://img.shields.io/badge/Paper-arXiv%3A2512.21857-9cf?style=for-the-badge" alt="Paper"></a>
</p>

本仓库是论文 [Fast Inference of Visual Autoregressive Model with Adjacency-Adaptive Dynamical Draft Trees](https://arxiv.org/abs/2512.21857) 的官方 PyTorch 实现。

自回归（AR）图像模型可以达到扩散级别的质量，但由于顺序推理的限制，生成一张 576x576 的图像大约需要 2,000 步。使用草稿树的推测解码可以加速 LLM，但在视觉 AR 模型上表现不佳，因为不同图像区域的 token 预测难度不同。我们发现将推测解码应用于视觉 AR 模型的关键障碍：由于不同图像区域预测难度不同，导致草稿树之间的接受率不一致。我们提出了 Adjacency-Adaptive Dynamical Draft Trees（ADT-Tree），一种邻接自适应动态草稿树，通过利用相邻 token 状态和先验接受率来动态调整草稿树的深度和宽度。ADT-Tree 通过水平邻接初始化，然后通过二分自适应优化深度/宽度，在简单区域生成更深的树，在复杂区域生成更宽的树。

所有主要代码参考了项目 [LANTERN](https://github.com/jadohu/LANTERN)

感谢 LANTERN 团队对开源社区的贡献

---

## 📰 新闻

- **[2025-11-28] TODO: 更换 eagle tree**
- **[2025-11-20] 🎉🎉🎉 ADT-Tree 已发布！🎉🎉🎉**
- **人工智能领域顶级会议论文门户: [CV_Paper_Portal](https://hongsong-wang.github.io/CV_Paper_Portal/)**

---

## 方法与性能

![方法](data/picture/method.png)

以下是不同方法的效果对比

![性能](data/picture/Performance.png)

---

## ⚙️ 安装

1. **安装依赖包**
    **环境要求**
    - Python >= 3.10
    - PyTorch >= 2.4.0
    
    安装 `requirements.txt` 中列出的依赖项。
    ```bash
    git clone https://github.com/Haodong-Lei-Ray/ADT-Tree.git
    cd ADT-Tree
    conda create -n ADT-Tree python=3.10 -y
    conda activate ADT-Tree
    pip install -r requirements.txt
    ```

2. **额外设置**
    1. **Lumina-mGPT**
        对于 [Lumina-mGPT](https://github.com/Alpha-VLLM/Lumina-mGPT)，需要安装 `flash_attention` 和 `xllmx` 包。
        ```bash
        pip install flash-attn --no-build-isolation
        cd models/base_models/lumina_mgpt
        pip install -e .
        ```

3. **模型权重**
    所有模型权重和其他所需数据应存放在 `ckpts/` 目录下。
    1. **Lumina-mGPT**
        对于 Lumina-mGPT，由于目前 transformers 中的 Chameleon 实现不包含 VQ-VAE 解码器，请手动下载 [Meta 提供的原始 VQ-VAE 权重](https://github.com/facebookresearch/chameleon) 并放置到以下目录：
        ```
        ckpts
        └── lumina_mgpt
            └── chameleon
                └── tokenizer
                    ├── text_tokenizer.json
                    ├── vqgan.yaml
                    └── vqgan.ckpt
        ```

        同时从 Huggingface 🤗 下载原始模型 [`Lumina-mGPT-7B-768`](https://huggingface.co/Alpha-VLLM/Lumina-mGPT-7B-768) 并放置到以下目录：
        ```
        ckpts
        └── lumina_mgpt
            └── Lumina-mGPT-7B-768
                ├── config.json
                ├── generation_config.json
                ├── model-00001-of-00002.safetensors
                └── other files...
        ```
    2. **Anole**
        对于 Anole，下载 [`Anole-7b-v0.1-hf`](https://huggingface.co/leloy/Anole-7b-v0.1-hf)，这是从 [`Anole`](https://huggingface.co/GAIR/Anole-7b-v0.1) 转换而来的 Huggingface 格式模型。
        
        此外，还需要下载 [Meta 提供的原始 VQ-VAE 权重](https://github.com/facebookresearch/chameleon) 并放置到以下目录：

        ```
        ckpts
        └── anole
            ├── Anole-7b-v0.1-hf
            |   ├── config.json
            |   ├── generation_config.json
            |   ├── model-00001-of-00003.safetensors
            |   └── other files...
            └── chameleon
                └── tokenizer
                    ├── text_tokenizer.json
                    ├── vqgan.yaml
                    └── vqgan.ckpt
        ```

        **（可选）训练好的 drafter**
        要使用训练好的 drafter，需要下载 [`anole_drafter`](https://huggingface.co/jadohu/anole_drafter) 并保存到 trained_drafters 目录下。
        ```
        ckpts
        └── anole
            └── trained_drafters
                └── anole_drafter
                    ├── config.json
                    ├── generation_config.json
                    ├── pytorch_model.bin
                    └── other files...
        ```

---

## ✨ 使用方法

### ANOLE
在 MSCOCO2017Val 上运行 ADT-Tree+LANTERN
```
cd ./ADT-Tree
prompt=MSCOCO2017Val
model=anole
temperature=1
model_type=eagle
lantern_delta=0.5
lantern_k=100

#output_path=/home/leihaodong/TIP26/exp/Anole/MSCOCO2017Val/lantern_ADT-Tree
output_path=<你的输出路径>

mkdir -p ${output_path}

nohup python main.py generate_images \
 --prompt $prompt \
 --model $model \
 --temperature $temperature \
 --model_type $model_type \
 --model_path leloy/Anole-7b-v0.1-hf \
 --drafter_path jadohu/anole_drafter \
 --output_dir $output_path \
 --lantern \
 --peanut \
 --lantern_k $lantern_k \
 --lantern_delta ${lantern_delta} \
 --num_images -1 > ${output_path}.log 2>&1 &

```


ADT-Tree+LANTERN


## ⚖️ 许可证

本项目遵循 Meta Platforms, Inc. 的 Chameleon 许可证分发。更多信息请参阅仓库中的 `LICENSE` 文件。

---

## 🙏 致谢

本仓库的构建广泛参考了 [FoundationVision/LlamaGen](https://github.com/FoundationVision/LlamaGen)、[Alpha-VLLM/Lumina-mGPT](https://github.com/Alpha-VLLM/Lumina-mGPT) 和 [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE)，借鉴了它们的许多核心组件和方法。

## Star History

<a href="https://www.star-history.com/?repos=Haodong-Lei-Ray%2FADT-Tree&type=date&legend=bottom-right">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=Haodong-Lei-Ray/ADT-Tree&type=date&theme=dark&legend=bottom-right" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=Haodong-Lei-Ray/ADT-Tree&type=date&legend=bottom-right" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=Haodong-Lei-Ray/ADT-Tree&type=date&legend=bottom-right" />
 </picture>
</a>


## 📄 引用

如果您使用了本代码库，或认为我们的工作有价值，请引用：

```
@misc{lei2025fastinferencevisualautoregressive,
      title={Fast Inference of Visual Autoregressive Model with Adjacency-Adaptive Dynamical Draft Trees}, 
      author={Haodong Lei and Hongsong Wang and Xin Geng and Liang Wang and Pan Zhou},
      year={2025},
      eprint={2512.21857},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.21857}, 
}
```
