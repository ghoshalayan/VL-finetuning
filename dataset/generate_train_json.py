import os
import json

# === Update this path to your dataset folder ===
folder_path = r"images"  # Folder where images are stored

# Allowed image extensions
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

data = []

for filename in os.listdir(folder_path):
    if filename.lower().endswith(image_extensions):
        image_path = f"dataset/images/{filename}"

        # Check if image is ECCO (by name convention)
        is_ecco = filename.lower().startswith("ecco_")

        # Generate JSON object
        entry = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": "Is this an ECCO shoe?"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Yes, this is an ECCO shoe."
                            if is_ecco
                            else "No, this is not an ECCO shoe.",
                        }
                    ],
                },
            ]
        }

        data.append(entry)

# Save JSON to train.json in the same folder (one level up from 'images')
output_path = os.path.join(os.path.dirname(folder_path), "train.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✔ train.json successfully created at: {output_path}")
