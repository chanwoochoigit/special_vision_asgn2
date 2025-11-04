# given 6 images from each dataset, create a set of 2x3 512x512 images and save it to a single image
import argparse
from pathlib import Path
from PIL import Image


def main():
    margin = 15  # Margin between images in pixels

    # WikiArt
    wikiart_indices = [1978, 1899]
    wikiart_fine_tuned_image_paths = [
        f"local_repo/WikiArt_ft/output/{i:05d}.png" for i in wikiart_indices
    ]
    wikiart_ground_truth_image_paths = [
        f"local_repo/WikiArt/input/test/{i:05d}_input.png" for i in wikiart_indices
    ]
    wikiart_raw_model_image_paths = [
        f"local_repo/WikiArt/output/{i:05d}.png" for i in wikiart_indices
    ]

    # Gundam
    gundam_indices = [1409, 1460]
    gundam_fine_tuned_image_paths = [
        f"local_repo/PixArt_ft/output/{i:05d}.png" for i in gundam_indices
    ]
    gundam_ground_truth_image_paths = [
        f"local_repo/PixArt/input/test/{i:05d}_input.png" for i in gundam_indices
    ]
    gundam_raw_model_image_paths = [
        f"local_repo/PixArt/output/{i:05d}.png" for i in gundam_indices
    ]

    # Calculate canvas size: 3 images + 2 margins horizontally, 2 images + 1 margin vertically
    canvas_width = 512 * 3 + margin * 2
    canvas_height = 512 * 2 + margin * 1

    wikiart_combined = Image.new("RGB", (canvas_width, canvas_height), color="white")
    wikiart_ft_images = [Image.open(img) for img in wikiart_fine_tuned_image_paths]
    wikiart_gt_images = [Image.open(img) for img in wikiart_ground_truth_image_paths]
    wikiart_raw_images = [Image.open(img) for img in wikiart_raw_model_image_paths]

    gundam_combined = Image.new("RGB", (canvas_width, canvas_height), color="white")
    gundam_ft_images = [Image.open(img) for img in gundam_fine_tuned_image_paths]
    gundam_gt_images = [Image.open(img) for img in gundam_ground_truth_image_paths]
    gundam_raw_images = [Image.open(img) for img in gundam_raw_model_image_paths]

    # image 1: 2 rows & each row has fine-tuned, ground truth, raw model
    # Top row
    wikiart_combined.paste(wikiart_ft_images[0], (0, 0))
    wikiart_combined.paste(wikiart_gt_images[0], (512 + margin, 0))
    wikiart_combined.paste(wikiart_raw_images[0], (512 * 2 + margin * 2, 0))
    # Bottom row
    wikiart_combined.paste(wikiart_ft_images[1], (0, 512 + margin))
    wikiart_combined.paste(wikiart_gt_images[1], (512 + margin, 512 + margin))
    wikiart_combined.paste(wikiart_raw_images[1], (512 * 2 + margin * 2, 512 + margin))

    # # image 2: 6 gundam images in 2 rows x 3 columns
    # Top row
    gundam_combined.paste(gundam_ft_images[0], (0, 0))
    gundam_combined.paste(gundam_gt_images[0], (512 + margin, 0))
    gundam_combined.paste(gundam_raw_images[0], (512 * 2 + margin * 2, 0))
    # Bottom row
    gundam_combined.paste(gundam_ft_images[1], (0, 512 + margin))
    gundam_combined.paste(gundam_gt_images[1], (512 + margin, 512 + margin))
    gundam_combined.paste(gundam_raw_images[1], (512 * 2 + margin * 2, 512 + margin))

    wikiart_combined.save("pixart_wikiart.png")
    gundam_combined.save("pixart_gundam.png")


if __name__ == "__main__":
    main()
