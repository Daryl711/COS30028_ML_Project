# Backbones: EfficientNet-B0, EfficientNet-B3, DINOv2 and BioCLIP

This note explains what a backbone is and how the four backbones in this project differ, then covers
BioCLIP vs CLIP, the zero-shot experiment and its showcase notebook.

## What a backbone is

Every model in this project has two parts:

```
image ──► BACKBONE ──► feature vector ──► crop head    ──► "tomato"
          (big, pretrained)               disease head ──► "early blight"
```

- The **backbone** turns an image into a list of numbers (a feature vector) that describes it:
  shapes, colours, textures, spots. It is large and pretrained on another dataset.
- The **heads** are small layers added on top. They learn to map those features to our crop and disease labels.

Swapping EfficientNet-B0 for B3 or DINOv2 changes only the backbone. The heads, loss and training loop
stay the same, which is why each folder in `baseline/MultiTaskBaseline/` is mostly the same files.

## The four backbones

| | EfficientNet-B0 | EfficientNet-B3 | DINOv2 ViT-S/14 | BioCLIP (ViT-B/16) |
|---|---|---|---|---|
| **Type** | CNN | CNN | Vision Transformer | Vision Transformer |
| **Size** | ~5M parameters | ~12M | ~22M | ~86M (image part) |
| **Image size used** | 224 px | 300 px | 224 px | 224 px |
| **Pretrained on** | ImageNet (~1.3M photos, with labels) | Same as B0 | ~142M images, **no labels** | ~10M photos of living things, with **species names** |
| **What it learned** | Recognise 1,000 everyday objects | Same, with more capacity | General visual features, by learning which image patches belong together | Match images to biology text (CLIP-style) |
| **In this project** | Trained baseline | Trained baseline | Trained baseline | Zero-shot test only so far |
| **Code** | `baseline/MultiTaskBaseline/InitialAttempt(Effb0)/` | `baseline/MultiTaskBaseline/EffB3/` | `baseline/MultiTaskBaseline/DINOv2/` | `baseline/ZeroShot/BioCLIP/` |

### CNN vs transformer

- A **CNN** (EfficientNet) slides small filters over the image, so it is naturally good at local detail:
  spots, lesions, edges.
- A **transformer** (DINOv2, BioCLIP) cuts the image into patches (14×14 or 16×16 px) and lets every patch
  look at every other patch. It is better at overall context ("brown spots spread across the whole leaf"),
  but usually needs more data or better pretraining to do well.

### EfficientNet-B0 vs B3

The same design at two sizes. B3 is deeper and wider and uses larger images (300 px), so it can see finer
lesion detail. The cost is speed and memory, which is why its config halves the batch size to 32.

### DINOv2

Pretrained **without labels**: it learned to give different crops of the same image similar features.
This gives very general features that tend to transfer well to new domains such as plant disease.
Its config uses a lower learning rate, and a smaller one again for earlier layers, so fine-tuning does
not erase what it learned.

### BioCLIP

The only one of the four that knows **biology and language**. It is OpenAI's CLIP trained further on
~10M images of organisms, each paired with its species name. Its image encoder already tells plant species
apart well, and its text encoder can score a crop-disease pair from its name, including pairs never seen in
training. It is also the largest of the four (about 4× DINOv2-S).

## Why this matters for the results

- **Seen accuracy** depends mainly on how good the backbone's features are, so bigger or better-pretrained
  backbones usually do better (roughly B0 < B3, DINOv2 < BioCLIP).
- **Unseen pairs** are the hard part. All three trained models use separate crop and disease heads, which
  have to guess combinations they never saw together. BioCLIP is the only backbone that could also use its
  text side to help with that.

For a fair comparison, everything except the backbone is kept the same: data split, heads, epochs and
evaluation. Any difference in results can then be put down to the backbone.

## BioCLIP and CLIP

### What CLIP is

CLIP (OpenAI, 2021) is an **image-text model**. It has two parts:

- an **image encoder** (a ViT-B/16) that turns an image into a vector
- a **text encoder** that turns a sentence into a vector in the same space

It was trained on ~400M image-caption pairs from the internet, so that each image's vector lands close to
its own caption's vector and far from the others. After training, "how well does this sentence describe
this image?" is just the similarity between the two vectors.

### What BioCLIP is

BioCLIP (Imageomics Institute, CVPR 2024) is **CLIP trained further on biology**. It uses the same design
and starts from CLIP's weights; what changes is the training data:

| | CLIP | BioCLIP |
|---|---|---|
| Model | ViT-B/16 image encoder + text encoder | Same (starts from CLIP's weights) |
| Training images | ~400M general web images (people, objects, scenes) | ~10M images of living things (TreeOfLife-10M: iNaturalist, Encyclopedia of Life, ...) |
| Training text | Ordinary web captions ("my dog at the beach") | Taxonomic and common names ("Plantae ... Solanaceae *Solanum lycopersicum*", "tomato") |
| Good at | Telling general things apart (dog vs car vs leaf) | Telling similar species apart (tomato vs potato leaf) |
| Size and speed | ~150M parameters, 224 px | Same, so one can replace the other directly |

The key idea is the **taxonomic text**. Because each name spells out the whole family tree, BioCLIP learns
that tomato and potato are close relatives (both Solanaceae) and that rice and wheat are both grasses.
That helps it with species it has rarely seen.

Neither model was trained on **disease** names, and BioCLIP's images are mostly whole organisms in nature
rather than close-ups of diseased leaves. So both should know crops better than diseases.

### Zero-shot classification

Both models can classify images **without any training on our data** ("zero-shot"):

1. Write a sentence for every class, e.g. `"a photo of a tomato leaf with early blight."`
   (we average three sentence templates per class).
2. Encode every sentence and every test image.
3. Each image gets the class whose sentence vector is most similar to its image vector.

Because a sentence can describe **any** crop-disease pair, pairs that never appear together in our
training set (the *unseen* test set) are scored the same way as seen ones. Our trained baselines cannot
do this: their crop and disease heads only learned from seen pairs.

The script is `baseline/ZeroShot/BioCLIP/zeroshot.py`. It reports the same metrics as the trained
baselines' `evaluate.py`, so the numbers can be compared directly.

## The showcase notebook: `bioclip_vs_clip_showcase.ipynb`

`baseline/ZeroShot/BioCLIP/bioclip_vs_clip_showcase.ipynb` is for **presenting** the zero-shot experiment. It answers one question: *does a model that already knows plants help on
our task, before we train anything?*

It is saved with all its outputs (tables, charts, example images), so it can be opened and shown
straight away without running anything. To run it again, open it on Colab (T4 GPU) or locally inside the
repo and choose Run all; the first run takes about 10 minutes.

| Section | What it shows |
|---|---|
| 1. What the two models are | The CLIP vs BioCLIP comparison and how zero-shot works |
| 2-3. Settings, code and data | Same Drive and repo setup as the training notebooks; works on Colab or locally |
| 4. The dataset | Dataset sizes and the 10 unseen crop-disease pairs |
| 5. The prompts | The exact sentences the models read for a seen, an unseen and a healthy pair |
| 6. Run the evaluation | Runs `zeroshot.py` for both models (skipped if results already exist) |
| 7. Results | Table and bar chart vs the trained baselines, and the seen/unseen trade-off curves |
| 8. Example predictions | 8 random test images with each model's guess, marked right or wrong |
| 9. Try a single image | Both models' top-5 guesses for one image; change `IMAGE` for a live demo |
| 10. Takeaways | What the results mean and the next step |

Trained EffB3 and DINOv2 results appear in the comparison automatically once their
`outputs/multitask_*/eval/metrics.json` files exist.

## Zero-shot results: BioCLIP vs CLIP

### How to read the numbers

- **Crop / disease**: accuracy of the crop or disease prediction on its own.
- **Pair**: the full crop + disease prediction, chosen among **all 333 pairs** (323 seen + 10 unseen).
- **HM** (harmonic mean) of seen and unseen pair accuracy: high only if **both** are high.
- Models tend to favour seen pairs. The standard fix is to add a bonus to unseen-pair scores.
  **Best HM** is the HM at the best bonus, and **AUC** is the area under the seen-vs-unseen accuracy curve
  over all bonuses (higher is better).
- **Chance level** is what random guessing would score: 1 in 55 crops, 1 in 175 diseases, 1 in 333 pairs.

### Results

Top-1 accuracy, BioCLIP and CLIP with no training, next to the trained EfficientNet-B0 baseline:

| | Seen crop | Seen disease | Seen pair | Unseen crop | Unseen disease | Unseen pair | HM | Best HM | AUC |
|---|---|---|---|---|---|---|---|---|---|
| BioCLIP, zero-shot | 0.349 | 0.028 | 0.061 | 0.325 | 0.257 | **0.274** | **0.100** | 0.105 | 0.036 |
| CLIP, zero-shot | 0.340 | 0.021 | 0.071 | 0.248 | 0.024 | 0.033 | 0.045 | 0.097 | 0.027 |
| EfficientNet-B0, trained | **0.900** | **0.694** | **0.679** | **0.669** | 0.108 | 0.049 | 0.091 | **0.607** | **0.471** |
| Random guessing | 0.018 | 0.006 | 0.003 | 0.018 | 0.006 | 0.003 | | | |

Pair accuracy on each of the 10 unseen pairs:

| Unseen pair | Test images | BioCLIP | CLIP |
|---|---|---|---|
| tomato, early blight | 472 | 0.383 | 0.021 |
| grape, black rot | 271 | 0.653 | 0.074 |
| bell pepper, bacterial spot | 220 | 0.009 | 0.018 |
| cotton, bacterial blight | 125 | 0.000 | 0.088 |
| mango, anthracnose | 95 | 0.042 | 0.000 |
| sugarcane, rust | 89 | 0.011 | 0.000 |
| cherry, aphids | 72 | 0.333 | 0.000 |
| cucumber, powdery mildew | 37 | 0.216 | 0.081 |
| corn, brown spot | 35 | 0.000 | 0.000 |
| cucumber, downy mildew | 32 | 0.000 | 0.000 |

### What the results indicate

1. **Both models already know crops.** About 35% seen crop accuracy with no training, roughly 20× chance.
   Pretraining on general images (CLIP) or on organisms (BioCLIP) gives useful plant features.

2. **Neither model knows diseases.** Seen disease accuracy is 2-3%, only slightly above chance.
   Disease names were not in either model's training text, so disease recognition has to be learned
   from our data. This is also why both score only 6-7% on seen pairs: the crop is often right, the
   disease almost never.

3. **On seen pairs, BioCLIP's biology training gives no advantage.** BioCLIP and CLIP are level on
   seen crops (35% vs 34%) and seen pairs (6% vs 7%, CLIP slightly ahead).

4. **On unseen pairs BioCLIP is far ahead, but the gain is narrow.** BioCLIP scores 27% vs CLIP's 3%,
   and its unseen crop accuracy is higher too (32% vs 25%). However, **358 of its 397 correct unseen
   predictions (90%) come from two pairs**: tomato early blight and grape black rot. It scores 0% on
   3 of the 10 unseen pairs and under 5% on 3 more. So BioCLIP recognises a few well-known diseases; it does not show a general
   ability to combine any crop with any disease. This also explains why its unseen disease accuracy (26%)
   is much higher than its seen disease accuracy (3%): the unseen set happens to contain those diseases.
   It is not explained by the image source: images from the same source dataset score 44% on tomato
   early blight but 0% on bell pepper bacterial spot.

5. **Zero-shot is far behind a trained model overall.** Trained EfficientNet-B0 reaches a best HM of 0.607
   and AUC 0.471; zero-shot BioCLIP reaches 0.105 and 0.036. The unseen-pair bonus barely helps the
   zero-shot models because their weak point is seen accuracy, not a bias towards seen pairs.

6. **Uncalibrated, BioCLIP beats the trained model on unseen pairs (27% vs 5%), but this is not a fair win.**
   The trained model is strongly biased towards the seen pairs it was trained on. Once the unseen-pair
   bonus is applied, it reaches 63% on unseen pairs, well above BioCLIP.

### What this means for the project

- **Zero-shot BioCLIP or CLIP is not a usable final model.** Training on our data is required,
  above all for diseases.
- **BioCLIP is still worth fine-tuning as a fourth backbone.** It starts with crop knowledge and some
  disease knowledge that CLIP lacks. Whether it beats EfficientNet-B3 or DINOv2 after fine-tuning is an
  open question that only training will answer.
- **Its text encoder is a possible way to improve unseen pairs.** When BioCLIP only has to choose among
  the 10 unseen pairs, it gets 65% right (CLIP 47%, chance 10%), so the text side does carry useful pair
  information. A method that fine-tunes the model while keeping the text encoder could build on this.
- **When reporting:** say that BioCLIP's unseen advantage is concentrated on two pairs, and that the
  unseen test set has only 10 pairs, half of whose images are tomato early blight and grape black rot.
