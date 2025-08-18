# Memory Augmented - OpenCLIP Codebase

This is the official repository of my Master's Thesis Project : ***Fine-grained image understanding with VLMs***

## Abstract
Vision-Language Models (VLM) have gained impressive generalization abilities, learning to
identify a vast range of concepts from web-scale data without direct supervision. A key limitation, however, is their difficulty with fine-grained image understanding, often failing to capture
the intricate details that define complex scenes. To address this shortcoming, we propose a
straightforward and efficient method for augmenting frozen foundation models with a persistent memory mechanism. By strategically replacing Multi-Layer Perceptron (MLP) sub-layers
in a Vision Transformer with trainable, key-value memory modules, we enhance the model’s
architectural capacity for detailed feature storage. A teacher-student knowledge distillation
framework is then employed to efficiently transfer knowledge from a pre-trained CLIP model
into our memory-enhanced student, eliminating the need for costly retraining from scratch.
Our results demonstrate that a memory-augmented vision encoder can be effectively trained to
achieve a new level of performance on long-caption fine-grained retrieval benchmarks. Moreover,
they highlight an important trade-off between specialization and generalization. Enhancing
fine-grained retrieval capabilities through this architectural modification can impact performance on pixel-level tasks like zero-shot semantic segmentation. These insights improve our
understanding of how architectural changes affect pre-trained VLM and provide a foundation
for future advancements in developing more comprehensive and efficient models for fine-grained
image understanding.
## Approach

| ![CLIP](https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/CLIP.png) |
|:--:|
| Image Credit: https://github.com/openai/CLIP |


## Data

To download datasets as webdataset, we recommend [img2dataset](https://github.com/rom1504/img2dataset).

## Training CLIP

### Install

We advise you first create a virtual environment with:

```
python3 -m venv .env
source .env/bin/activate
pip install -U pip
```

You can then install openclip for training with `pip install 'open_clip_torch[training]'`.


### Sample single-process running code:

```bash
python -m open_clip_train.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/path/to/train_data.csv"  \
    --val-data="/path/to/validation_data.csv"  \
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val=/path/to/imagenet/root/val/ \
    --warmup 10000 \
    --batch-size=128 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs=30 \
    --workers=8 \
    --model RN50
```


#### Single-Node

We make use of `torchrun` to launch distributed jobs. The following launches a
a job on a node of 4 GPUs:

```bash
cd open_clip/src
torchrun --nproc_per_node 4 -m open_clip_train.main \
    --train-data '/data/cc12m/cc12m-train-{0000..2175}.tar' \
    --train-num-samples 10968539 \
    --dataset-type webdataset \
    --batch-size 320 \
    --precision amp \
    --workers 4 \
    --imagenet-val /data/imagenet/validation/
```
#### Multi-Node

The same script above works, so long as users include information about the number
of nodes and host node.

```bash
cd open_clip/src
torchrun --nproc_per_node=4 \
    --rdzv_endpoint=$HOSTE_NODE_ADDR \
    -m open_clip_train.main \
    --train-data '/data/cc12m/cc12m-train-{0000..2175}.tar' \
    --train-num-samples 10968539 \
    --dataset-type webdataset \
    --batch-size 320 \
    --precision amp \
    --workers 4 \
    --imagenet-val /data/imagenet/validation/
```

## Acknowledgements 

Current development of this repository is based on [CLIP](https://github.com/openai/CLIP). Moreover, several training, distillation and evaluation code has relied on the existing the work of [TULIP](https://github.com/ivonajdenkoska/tulip) & [FLAIR](https://github.com/ExplainableML/flair)
