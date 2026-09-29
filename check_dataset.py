from pathlib import Path

train_path = Path(r"C:\Users\user\OneDrive\Desktop\dataset\PlantVillage\train")
val_path = Path(r"C:\Users\user\OneDrive\Desktop\dataset\PlantVillage\val")

extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

selected_classes = [
    "Tomato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",

    "Pepper,_bell___healthy",
    "Pepper,_bell___Bacterial_spot",

    "Potato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",

    "Corn_(maize)___healthy",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"
]


def count_images(folder):
    return sum(
        1
        for file in folder.rglob("*")
        if file.is_file() and file.suffix in extensions
    )


print("TRAIN DATA")
print("=" * 50)

for class_name in selected_classes:
    folder = train_path / class_name

    if folder.exists():
        print(f"{class_name}: {count_images(folder)}")
    else:
        print(f"{class_name}: NOT FOUND")


print()
print("VALIDATION DATA")
print("=" * 50)

for class_name in selected_classes:
    folder = val_path / class_name

    if folder.exists():
        print(f"{class_name}: {count_images(folder)}")
    else:
        print(f"{class_name}: NOT FOUND")