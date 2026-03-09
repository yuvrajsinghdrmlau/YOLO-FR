import os
import cv2
import torch
import numpy as np
from models.yolofr_pytorch import YOLOFR

# -------------------------
# PATHS
# -------------------------
IMAGE_DIR = r"PyTorch_Wheels\yolov5\Dataset_for_yolofr\Test_images"
OUTPUT_DIR = r"PyTorch_Wheels\yolov5\ataset_for_yolofr\output_images"
MODEL_PATH = r"PyTorch_Wheels\yolov5\yolofr_test.pth"

IMG_SIZE = 320
CONF_THRESH = 0.25
IOU_THRESH = 0.45

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# LOAD MODEL
# -------------------------
print("Loading model...")
model = YOLOFR(num_classes=1)
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()
print("Model loaded.")

# -------------------------
# FAST NMS
# -------------------------
def fast_nms(pred, conf_thres=0.25, iou_thres=0.45):
    pred = pred[pred[:,4] > conf_thres]
    if len(pred) == 0:
        return []

    boxes = pred[:, :4]
    scores = pred[:, 4]

    keep = torch.ops.torchvision.nms(boxes, scores, iou_thres)
    return pred[keep]

# -------------------------
# RUN ON ALL IMAGES
# -------------------------
files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".png"))]

for fname in files:
    path = os.path.join(IMAGE_DIR, fname)
    image = cv2.imread(path)
    if image is None:
        continue

    h, w = image.shape[:2]

    # Preprocess
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = img[:, :, ::-1] / 255.0
    img = torch.tensor(img).float().permute(2,0,1).unsqueeze(0)

    with torch.no_grad():
        p_small, p_mid, p_large = model(img)

    preds = torch.cat([
        p_small.view(-1, 6),
        p_mid.view(-1, 6),
        p_large.view(-1, 6)
    ], dim=0)

    final_boxes = fast_nms(preds, CONF_THRESH, IOU_THRESH)

    # Draw detections
    for det in final_boxes:
        x1, y1, x2, y2, conf, cls = det

        x1 = int((x1 / IMG_SIZE) * w)
        y1 = int((y1 / IMG_SIZE) * h)
        x2 = int((x2 / IMG_SIZE) * w)
        y2 = int((y2 / IMG_SIZE) * h)

        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(image, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Save output
    out_path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(out_path, image)
    print("Saved:", out_path)

print("DONE! All images processed.")
