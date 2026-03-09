# check_dataset.py
import os, sys

base = "Total_mergedataset"
imgs_train = os.path.join(base, "images", "train")
imgs_val   = os.path.join(base, "images", "val")
lbls_train = os.path.join(base, "labels", "train")
lbls_val   = os.path.join(base, "labels", "val")

def files(path, exts):
    return [f for f in os.listdir(path) if f.lower().endswith(exts)]

def check_pair(img_dir, lbl_dir):
    imgs = files(img_dir, ('.jpg','.jpeg','.png'))
    lbls = files(lbl_dir, ('.txt',))
    imgs_base = set(os.path.splitext(f)[0] for f in imgs)
    lbls_base = set(os.path.splitext(f)[0] for f in lbls)
    missing_lbl = sorted(list(imgs_base - lbls_base))[:10]
    missing_img = sorted(list(lbls_base - imgs_base))[:10]
    return len(imgs), len(lbls), missing_lbl, missing_img, imgs[:5], lbls[:5]

for (i, (imgd, lbld)) in enumerate(((imgs_train, lbls_train),(imgs_val, lbls_val))):
    ni, nl, misslbl, missimg, simgs, slbls = check_pair(imgd, lbld)
    print(f"\nFolder {i+1}:")
    print(f"  Images: {ni}, Labels: {nl}")
    if misslbl:
        print("  Examples of images missing labels:", misslbl)
    if missimg:
        print("  Examples of labels without images:", missimg)
    print("  Sample images:", simgs)
    print("  Sample labels:", slbls)

# Count unique classes across all label files
classes = set()
for sub in ("train","val"):
    lbl_path = os.path.join(base, "labels", sub)
    for f in os.listdir(lbl_path):
        if f.lower().endswith('.txt'):
            for line in open(os.path.join(lbl_path, f), 'r', encoding='utf-8'):
                line=line.strip()
                if not line: continue
                parts = line.split()
                classes.add(parts[0])
print("\nUnique class IDs found in labels:", sorted(classes))
print("Number of unique classes:", len(classes))
