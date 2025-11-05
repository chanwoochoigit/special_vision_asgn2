Comparison of vanilla and fine-tuned (LoRA) PixArt model performances
Datasets used:
1. Gundam
2. WikiArt 

run:
```bash
# complete pipeline from image prep, fine-tuning and inference
bash scripts/pixart_scripts/wikiart/fine_tune_wikiart.sh

# create sample images for qualitative inspection
python scripts/pixart_scripts/common/create_qualitative_images.py

# record comprehensive metrics for comparison
bash scripts/pixart_scripts/gundam/compute_metrics_gundam.sh
bash scripts/pixart_scripts/wikiart/compute_metrics_wikiart.sh
```