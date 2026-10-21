# Fine-tuning the encoder end to end

Script: `compat/finetune.py`. Configs: `compat/configs/finetune_method.yaml` (ResNet-18, papers' setup)
and `compat/configs/finetune_app.yaml` (best fashion encoder, last blocks unfrozen).
Colab: `finetune_colab.ipynb`. Duke cluster: `scripts/finetune_dcc.sbatch`.

Phase 1 trained small heads on **frozen** image features. Here the image encoder itself learns too, so
it can pick up what matters for "goes together" (silhouette, formality, material) instead of only
what ImageNet or CLIP pretraining cared about. Every published Polyvore result does this, and it's
where most of their gains over frozen features came from.

## The choices, in plain terms

**Loss: InfoNCE with in-batch same-category negatives and a learned temperature.** It won Phase 1 on
both encoders (ResNet-18: 0.822 → 0.847 test AUC; FashionCLIP: 0.915 → 0.921). Each anchor picks its
real partner out of all same-category items in the batch, which is a many-way version of the FITB
question. It also works on item *pairs*, so a step encodes 2 × batch images; outfit-level losses
would encode about 20 images per example.

**Two learning rates: small for the encoder, large for the head.** The head starts from random
weights and has to learn from scratch (1e-3). The encoder already knows a lot. Large updates would
destroy that ("catastrophic forgetting"), so it gets 10× smaller steps for ResNet-18 (1e-4) and 100×
smaller for the ViT (1e-5), because big pretrained transformers are more fragile.

**Encoder frozen for the first epoch (head-only warmup).** At step 0 the head is random, so the
gradients it sends back into the encoder are noise. Letting the head learn on fixed features first
means that by the time the encoder starts moving, it receives meaningful gradients. While frozen, the
encoder also runs in eval mode, so ResNet's BatchNorm statistics don't drift.

**Linear warmup, then cosine decay.** When the encoder unfreezes, its learning rate ramps from 0 to
full over one epoch, which avoids a shock to the pretrained weights. Then both learning rates follow
a cosine curve down to 0: big steps early while there's a lot to learn, tiny steps at the end to
settle into a minimum instead of bouncing around it.

**Partial unfreezing for the app encoder (last 4 of 12 blocks).** Early transformer blocks learn
generic visual features (edges, textures); late blocks learn task-specific combinations. Training only
the last few adapts what needs adapting, keeps the rest of the pretrained knowledge, uses much less
memory (no gradients or optimizer state for the frozen blocks), and overfits less with only 17k
training outfits. ResNet-18 is small, so the method config trains every layer, matching the papers.

**AdamW with weight decay 0.05 on the ViT.** AdamW applies weight decay directly to the weights
instead of mixing it into the gradient (which plain Adam effectively does). 0.05 is the standard
value for fine-tuning ViTs.

**Mixed precision (CUDA only).** Matrix maths runs in 16-bit, which is about 2× faster and uses half
the activation memory, while the master weights stay in 32-bit. On an A100 it uses **bf16**, which
has float32's range, so nothing overflows. A T4 has no bf16, so it uses **fp16 with a GradScaler**:
the loss is multiplied by a large factor before the backward pass so small gradients don't round to
zero in fp16, then the gradients are divided back before the update.

**Light augmentation: random resized crop + horizontal flip, no color changes.** Crops and flips
teach the encoder that a jacket is the same jacket slightly zoomed or mirrored, which helps when
fine-tuning on only 17k outfits. Color jitter is deliberately left out: color is one of the strongest
compatibility signals ("navy and white"), and changing it would teach the model to ignore it.

**The largest batch that fits.** For InfoNCE the batch *is* the set of negatives, so a bigger batch
means a harder, more informative multiple-choice question per anchor. `--find-batch` halves the batch
from the config value until a forward + backward pass fits on the GPU you got.

**Gradient checkpointing (app config).** Normally every layer's activations are kept for the backward
pass. Checkpointing keeps only a few and recomputes the rest during backward: about 30% slower, but
far less memory, so a bigger batch fits on a 16 GB T4.

**Checkpoints every 200 steps and every epoch, with exact resume.** Colab disconnects without
warning. The checkpoint holds the model, optimizer, GradScaler, epoch and step, the best validation
score so far and every random-number-generator state. Each epoch's training pairs come from an RNG
seeded by (seed, epoch), so a resumed epoch replays the same pairs and skips the batches already
done. Files are written to a temporary name and renamed, so a disconnect mid-save can't leave a
corrupt checkpoint.

**Early stopping on validation AUC; test once.** After every epoch the script encodes all validation
items and computes compatibility AUC and FITB exactly as in Phase 1. If AUC hasn't improved for
`patience` epochs it stops, reloads the best epoch's weights, and evaluates the test set **once**.

**Fixed seeds.** Python, NumPy and torch are seeded, and each epoch's pairs come from a seeded
generator. GPU kernels (cuDNN autotuning, atomics) aren't bit-deterministic, so expect run-to-run
differences around ±0.002 AUC. That's why final results use 3 seeds (`--set seed=1`, `seed=2`).

**"Epoch" = 131k training pairs, not all 386k.** It's still about 262k images. Shorter epochs mean
validation, checkpointing and early-stopping decisions every few minutes rather than every half hour.

## Running on Google Colab

1. Push the branch from your Mac: `git push origin compat-v2`.
2. Open `finetune_colab.ipynb` in Colab (File → Upload notebook, or open it from GitHub).
3. Runtime → Change runtime type → **T4 GPU** (free) or **A100** (Colab Pro).
4. In the settings cell set `CONFIG = "method"` or `"app"`.
5. Runtime → **Run all**. You'll be asked for:
   - a **GitHub token** with read access to `Vikhram-123/sweep` (fine-grained token → Contents: Read-only);
   - the first time only, a **Hugging Face read token** for the gated dataset. The data is then cached
     in `MyDrive/sweep/data`.
6. Watch the log: `img/s` per step, then validation AUC/FITB after each epoch.
7. If Colab disconnects, run all again. It resumes from `MyDrive/sweep/checkpoints/<name>/last.pt`.
8. For seeds 1 and 2, add `--set seed=1 name=ft_method_resnet18_seed1` to the train cell. Give each seed
   its own name, so each gets its own checkpoint folder.
9. When it finishes, copy `MyDrive/sweep/results/<name>/` into `results/runs/` on your Mac and commit.

## Running on the Duke Compute Cluster (Slurm)

Facts from dcc.duke.edu: GPU partition `gpu-common` (RTX 2080 Ti 11 GB, RTX A5000 24 GB, RTX 5000 Ada
32 GB), max walltime 2 days, no preemption. `/work/<netid>` is for datasets and results but is
**purged after 75 days of no modification**. Accounts that are only on the DCC through a course
can't use `gpu-common`.

One-time setup:

```bash
ssh <netid>@dcc-login.oit.duke.edu

# Python environment in your home directory (25 GB quota)
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/etc/profile.d/conda.sh
conda create -y -n sweep python=3.11
conda activate sweep
pip install torch torchvision                      # CUDA wheels from PyPI include the CUDA runtime
pip install transformers open_clip_torch pyarrow pyyaml scikit-learn matplotlib huggingface_hub pillow

# Code (git asks for username + token; nothing is stored unless you set a credential helper)
cd /work/$USER
git clone --branch compat-v2 https://github.com/Vikhram-123/sweep.git
cd sweep
```

Data, either way:

```bash
# (a) download on the cluster (asks for the HF token without echoing it)
read -s -p "HF token: " HF_TOKEN && export HF_TOKEN && python -m compat.download --out data/polyvore; unset HF_TOKEN
# (b) or copy from your Mac. --no-times gives files fresh timestamps so the 75-day purge doesn't
#     delete them immediately (the parquet files keep their 2026 download dates otherwise)
rsync -a --no-times ~/Downloads/Projects/Sweep/data/polyvore/ <netid>@dcc-login.oit.duke.edu:/work/<netid>/sweep/data/polyvore/
```

Check that it runs, then submit:

```bash
srun -p gpu-common --gres=gpu:1 --mem=16G --time=00:30:00 --pty bash -i   # interactive GPU shell
conda activate sweep && cd /work/$USER/sweep
python -m compat.finetune compat/configs/finetune_method.yaml --smoke
python -m compat.finetune compat/configs/finetune_method.yaml --find-batch
exit

sbatch scripts/finetune_dcc.sbatch method        # or: app
squeue -u $USER                                  # queue status
tail -f logs/ft-sweep-ft-<jobid>.out             # progress
# If the job hits its time limit, sbatch the same line again: it resumes from /work/$USER/sweep_ckpt/<name>.
```

Copy results back with `rsync -a --no-times <netid>@dcc-login.oit.duke.edu:/work/<netid>/sweep/results/runs/<name> results/runs/`.

## How long an epoch takes (estimates)

One epoch = 131k pairs = 262k training images, plus encoding 14.6k validation images. Each epoch is
limited by whichever is slower: **decoding JPEGs on the CPU** or **the GPU's compute**.

- **CPU side:** measured on one M1 Pro core: 926 img/s at 112 px, 499 at 224, 209 at 384 (decode,
  crop, flip, normalize). Assumes a Colab vCPU is about 0.55× an M1 core, so a T4 runtime (2 vCPUs)
  manages about 1.1× one M1 core, and an A100 runtime (12 vCPUs) about 6.5×.
- **GPU side:** training FLOPs per image (forward, plus backward through the trained layers, plus the
  checkpointing recompute), divided by realistic sustained throughput: about 18 TFLOP/s for a T4 in
  fp16 and about 120 for an A100 in bf16 (roughly 25–40% of peak).

| config | train FLOPs / img | T4: CPU-bound | T4: GPU-bound | **T4 epoch** | A100: CPU | A100: GPU | **A100 epoch** |
|---|---|---|---|---|---|---|---|
| method: ResNet-18, 112 px, all layers | ~2 GFLOP | ~4.3 min | ~0.5 min | **~5 min** | ~0.7 min | <0.1 min | **~1 min** |
| app: FashionCLIP ViT-B/32, 224, last 4 blocks | ~18 GFLOP | ~8 min | ~4.5 min | **~9 min** | ~1.4 min | ~0.7 min | **~1.5–2 min** |
| app: Marqo-FashionSigLIP ViT-B/16, 224, last 4 | ~70 GFLOP | ~8 min | ~17 min | **~18 min** | ~1.4 min | ~2.5 min | **~3 min** |
| app: ZooClaw-FashionSigLIP2 ViT-B/16, 384, last 4 | ~245 GFLOP | ~19 min | ~60 min | **~60 min** | ~3.3 min | ~9 min | **~9–10 min** |

Treat these as ±50%. Real numbers depend on the exact Colab machine, its disk and its CPUs. The
training log prints `img/s` every 50 steps: epoch time ≈ 262,000 / that number, plus about a minute
for validation. With early stopping most runs end within 8–15 epochs, so a method run is about an
hour on a T4 and 15 minutes on an A100. On the Duke cluster an RTX A5000 sits between the two (about
2–3× a T4) and gets 8 CPUs from the sbatch script, so the method config should be GPU-light and
CPU-bound at about 2 min per epoch.

## Interview questions

1. *Why different learning rates for the encoder and the head?* The head is random and must learn
   fast; the encoder is pretrained and large updates would erase what it knows.
2. *Why freeze the encoder at first?* A random head sends noise gradients into the encoder. Training
   the head first means the encoder's first updates are useful ones.
3. *Why no color augmentation?* Color is one of the main compatibility signals; jittering it would
   teach the model to ignore exactly what matters.
4. *Why fp16 needs a GradScaler but bf16 doesn't.* fp16 has a narrow exponent range, so small gradients
   underflow to zero unless the loss is scaled up. bf16 keeps float32's exponent range.
5. *How does resume reproduce the same epoch?* Pairs come from an RNG seeded by (seed, epoch), and the
   checkpoint records the step, so a resumed run regenerates the epoch and skips what's done.
6. *Why does InfoNCE want a big batch?* The other same-category items in the batch are the negatives.
   More of them make each softmax question harder and more informative.
