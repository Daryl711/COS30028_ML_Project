How to run the baseline:



Multitask:

python baseline/train.py





Independent baseline:

python baseline/train.py --mode crop    --experiment indep\_crop\_effb0

python baseline/train.py --mode disease --experiment indep\_disease\_effb0





to let the training uses gpu:

python -m pip install --force-reinstall --no-deps torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126





Evaluate:

multitask:

python baseline/evaluate.py --checkpoints outputs/multitask\_effb0/best.pt --out outputs/multitask\_effb0/eval



Independent:

python baseline/evaluate.py --checkpoints outputs/indep\_crop\_effb0/best.pt outputs/indep\_disease\_effb0/best.pt --out outputs/indep\_effb0/eval



&#x09;

|File|What it shows|
|-|-|
|confusion\_seen\_crops.png|All 55 crops. Each cell shows a percentage, empty cells are blank, and each row label includes its number of test images|
|confusion\_seen\_crops\_worst.png|A zoom on the 15 crops with the lowest accuracy, among crops with at least 5 test images. Each cell shows a percentage and an image count.|
|top\_confusions\_seen.png|The 20 most common mistakes, such as "paddy \| healthy → blast: 20 of 32 images". This replaces the 323×333 pair heatmap, which couldn't be read at any size.|





How the model works (baseline/model.py)



image (224×224) → EfficientNet-B0 backbone → 1280-number feature vector

&#x20;                                           ├─ Dropout → Linear → crop logits (55 crops)

&#x20;                                           └─ Dropout → Linear → disease logits



\- Backbone: timm.create\_model("efficientnet\_b0", pretrained=True, num\_classes=0). Setting num\_classes=0 removes timm's own classifier, so the backbone returns just the pooled features. Stochastic depth (drop\_path\_rate=0.2) adds regularization.

\- Heads: an nn.ModuleDict holding one dropout (0.3) + linear layer per task.

\- Loss: MultiTaskLoss = 1.0 × crop cross-entropy + 1.0 × disease cross-entropy, with label smoothing of 0.1.

\- The same class also builds the independent baseline: --mode crop or --mode disease gives a model with only one head.



Data pipeline (baseline/dataset.py)



\- Input: three CSVs for the DPD-Small dataset (train, test\_seen, test\_unseen). Each row holds the image path, crop label, disease label and composition label.

\- Validation split: 10% of the training data is held out within each crop+disease pair, used to pick the best epoch. The test sets are never used during training.

\- Training augmentation: random resized crop, horizontal and vertical flips, TrivialAugmentWide, ImageNet normalization and random erasing. These aim at the gap between lab photos and field photos.

\- Eval transform: resize, then center-crop to 224.



Training (baseline/train.py)



\- Optimizer: AdamW with two learning rates: 2e-4 for the pretrained backbone and 1e-3 for the new heads.

\- Schedule: 2 epochs of linear warmup, then cosine decay. It steps once per batch.

\- Speed: mixed precision (torch.amp + GradScaler) when a CUDA GPU is present.

\- Run length: up to 30 epochs at batch size 64. Training stops early after 8 epochs with no improvement in the validation score (mean top-1 accuracy of the two heads).

\- Outputs go to outputs/multitask\_effb0/: best.pt, last.pt, history.csv and a copy of the config used.



Evaluation (baseline/evaluate.py, metrics.py)



\- Reports top-1 and top-5 accuracy for crop, disease and the full crop+disease pair, on both seen and unseen test images.

\- Scoring a pair: log p(crop) + log p(disease), computed only over valid pairs. This lets two separate heads rank a combination they never saw together in training.

\- Harmonic mean: combines seen and unseen accuracy into one score.

\- Calibration sweep: adds a bias to the scores of unseen pairs to offset the model's lean toward pairs it has seen. This is standard in compositional zero-shot papers.



report:

result - fair comparison

indicate observation

explanation - reasoning what can be improved

Quantitative and Qualitative

