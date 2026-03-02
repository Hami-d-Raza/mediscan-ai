import sys, os, glob
sys.path.insert(0, '.')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from app.services.image_model import predict_image

LABEL_MAP = {
    "Glioma": "glioma",
    "Meningioma": "meningioma",
    "Normal (No Tumor)": "notumor",
    "Pituitary Tumor": "pituitary",
}

imgs = sorted(glob.glob("data/brain_tumor_mri/Testing/glioma/*"))
print(f"Total glioma test images: {len(imgs)}\n")

counts = {"glioma": 0, "meningioma": 0, "notumor": 0, "pituitary": 0}
for img in imgs:
    result = predict_image(img)
    pred_folder = LABEL_MAP.get(result["prediction"], "?")
    counts[pred_folder] = counts.get(pred_folder, 0) + 1

total = len(imgs)
print("Results for all GLIOMA test images:")
print(f"  Correctly predicted Glioma       : {counts['glioma']}/{total} = {counts['glioma']/total*100:.1f}%")
print(f"  Wrong -> Meningioma              : {counts['meningioma']}/{total} = {counts['meningioma']/total*100:.1f}%")
print(f"  Wrong -> Normal (No Tumor)       : {counts['notumor']}/{total} = {counts['notumor']/total*100:.1f}%")
print(f"  Wrong -> Pituitary               : {counts['pituitary']}/{total} = {counts['pituitary']/total*100:.1f}%")

# Also show all 4 classes overall accuracy
print("\n--- Overall per-class accuracy ---")
for cls in ["glioma", "meningioma", "notumor", "pituitary"]:
    class_imgs = sorted(glob.glob(f"data/brain_tumor_mri/Testing/{cls}/*"))
    correct = 0
    for img in class_imgs:
        result = predict_image(img)
        if LABEL_MAP.get(result["prediction"]) == cls:
            correct += 1
    n = len(class_imgs)
    print(f"  {cls:<12}: {correct}/{n} = {correct/n*100:.1f}%")
