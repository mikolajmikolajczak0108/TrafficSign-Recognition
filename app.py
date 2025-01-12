import os
import cv2
import csv
import shutil
import random
import logging
import numpy as np
import sys
import subprocess
from flask import Flask, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
import torch
import yt_dlp
import albumentations as A
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads/"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# -----------------------------
# MAPPING - 43 classes GTSRB
# -----------------------------
sign_names = {
    "0": "Speed limit (20km/h)",
    "1": "Speed limit (30km/h)",
    "2": "Speed limit (50km/h)",
    "3": "Speed limit (60km/h)",
    "4": "Speed limit (70km/h)",
    "5": "Speed limit (80km/h)",
    "6": "End of speed limit (80km/h)",
    "7": "Speed limit (100km/h)",
    "8": "Speed limit (120km/h)",
    "9": "No passing",
    "10": "No passing veh over 3.5 tons",
    "11": "Right-of-way at intersection",
    "12": "Priority road",
    "13": "Yield",
    "14": "Stop",
    "15": "No vehicles",
    "16": "Veh > 3.5 tons prohibited",
    "17": "No entry",
    "18": "General caution",
    "19": "Dangerous curve left",
    "20": "Dangerous curve right",
    "21": "Double curve",
    "22": "Bumpy road",
    "23": "Slippery road",
    "24": "Road narrows on the right",
    "25": "Road work",
    "26": "Traffic signals",
    "27": "Pedestrians",
    "28": "Children crossing",
    "29": "Bicycles crossing",
    "30": "Beware of ice/snow",
    "31": "Wild animals crossing",
    "32": "End speed + passing limits",
    "33": "Turn right ahead",
    "34": "Turn left ahead",
    "35": "Ahead only",
    "36": "Go straight or right",
    "37": "Go straight or left",
    "38": "Keep right",
    "39": "Keep left",
    "40": "Roundabout mandatory",
    "41": "End of no passing",
    "42": "End no passing veh > 3.5 tons",
}

TRAIN_DIR = "GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"

# YOLO
yolo_model = None
yolo_video_capture = None
yolo_video_active = False


@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------------------------------------
# DOWNLOAD YOLOv5
# -----------------------------------------------------------
@app.route("/download_assets", methods=["GET", "POST"])
def download_assets():
    if request.method == "POST":
        steps = 2
        with tqdm(
            total=steps, desc="Pobieranie YOLOv5 i instalowanie zależności", unit="krok"
        ) as pbar:
            # 1) Clone YOLOv5 if not existing
            if not os.path.exists("yolov5"):
                subprocess.run(
                    ["git", "clone", "https://github.com/ultralytics/yolov5.git"],
                    check=True,
                )
            pbar.update(1)
            # 2) Install YOLOv5 requirements
            req_file = os.path.join("yolov5", "requirements.txt")
            if os.path.exists(req_file):
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", req_file], check=True
                )
            else:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "ultralytics"], check=True
                )
            pbar.update(1)
        return render_template(
            "download_assets.html", message="Pobrano YOLOv5 i zainstalowano zależności!"
        )
    else:
        return render_template("download_assets.html", message=None)


# -----------------------------------------------------------
# CONVERT GTSRB -> YOLO
# -----------------------------------------------------------
def convert_gtsrb_to_yolo():
    if os.path.exists("my_gtsrb_yolo"):
        shutil.rmtree("my_gtsrb_yolo")
    os.makedirs("my_gtsrb_yolo/images/train", exist_ok=True)
    os.makedirs("my_gtsrb_yolo/images/val", exist_ok=True)
    os.makedirs("my_gtsrb_yolo/labels/train", exist_ok=True)
    os.makedirs("my_gtsrb_yolo/labels/val", exist_ok=True)

    data_all = []
    class_dirs = sorted(os.listdir(TRAIN_DIR))

    for class_id_str in tqdm(class_dirs, desc="Odczytywanie klas"):
        class_dir = os.path.join(TRAIN_DIR, class_id_str)
        if not os.path.isdir(class_dir):
            continue
        csv_path = os.path.join(class_dir, f"GT-{class_id_str}.csv")
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                filename = row["Filename"]
                x1 = int(row["Roi.X1"])
                y1 = int(row["Roi.Y1"])
                x2 = int(row["Roi.X2"])
                y2 = int(row["Roi.Y2"])
                cls_id = int(row["ClassId"])
                img_path = os.path.join(class_dir, filename)
                if not os.path.exists(img_path):
                    continue
                img = cv2.imread(img_path)
                if img is None:
                    continue
                h_ori, w_ori, _ = img.shape
                bb_width = x2 - x1
                bb_height = y2 - y1
                x_center = x1 + bb_width / 2.0
                y_center = y1 + bb_height / 2.0
                x_center_norm = x_center / w_ori
                y_center_norm = y_center / h_ori
                w_norm = bb_width / w_ori
                h_norm = bb_height / h_ori
                data_all.append(
                    (img_path, cls_id, x_center_norm, y_center_norm, w_norm, h_norm)
                )

    random.shuffle(data_all)
    split_idx = int(len(data_all) * 0.8)
    train_data = data_all[:split_idx]
    val_data = data_all[split_idx:]

    for i, (img_path, cls_id, x_c, y_c, w, h) in tqdm(
        enumerate(train_data), total=len(train_data), desc="Przetwarzanie danych TRAIN"
    ):
        base_name = f"train_{i:06d}"
        img = cv2.imread(img_path)
        if img is None:
            continue
        out_img_path = os.path.join(
            "my_gtsrb_yolo", "images", "train", base_name + ".jpg"
        )
        cv2.imwrite(out_img_path, img)
        out_txt_path = os.path.join(
            "my_gtsrb_yolo", "labels", "train", base_name + ".txt"
        )
        with open(out_txt_path, "w") as ftxt:
            ftxt.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    for i, (img_path, cls_id, x_c, y_c, w, h) in tqdm(
        enumerate(val_data), total=len(val_data), desc="Przetwarzanie danych VAL"
    ):
        base_name = f"val_{i:06d}"
        img = cv2.imread(img_path)
        if img is None:
            continue
        out_img_path = os.path.join(
            "my_gtsrb_yolo", "images", "val", base_name + ".jpg"
        )
        cv2.imwrite(out_img_path, img)
        out_txt_path = os.path.join(
            "my_gtsrb_yolo", "labels", "val", base_name + ".txt"
        )
        with open(out_txt_path, "w") as ftxt:
            ftxt.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

    yolo_yaml = """train: my_gtsrb_yolo/images/train
val: my_gtsrb_yolo/images/val

names:
"""
    for i in range(len(sign_names)):
        yolo_yaml += f"  - '{sign_names[str(i)]}'\n"

    with open("gtsrb.yaml", "w") as fy:
        fy.write(yolo_yaml)

    print("Konwersja GTSRB -> YOLO zakończona. Plik gtsrb.yaml utworzony.")


@app.route("/convert_gtsrb_to_yolo", methods=["GET", "POST"])
def convert_gtsrb_to_yolo_route():
    if request.method == "POST":
        try:
            convert_gtsrb_to_yolo()
            return render_template(
                "convert_gtsrb.html", message="Konwersja zakończona sukcesem!"
            )
        except Exception as e:
            return render_template("convert_gtsrb.html", message=f"Błąd: {e}")
    else:
        return render_template("convert_gtsrb.html", message=None)


# -----------------------------------------------------------
# TRAIN YOLO
# -----------------------------------------------------------
@app.route("/train_yolo", methods=["GET", "POST"])
def train_yolo():
    if request.method == "POST":
        # Możemy dodać prosty pasek postępu, ale to tylko jeden krok - uruchomienie subprocess
        print("Rozpoczynamy trening YOLOv5...")
        try:
            with tqdm(total=1, desc="Trening YOLOv5", unit="krok") as pbar:
                cmd = [
                    sys.executable,
                    "yolov5/train.py",
                    "--img",
                    "640",
                    "--batch",
                    "16",
                    "--epochs",
                    "5",
                    "--data",
                    "gtsrb.yaml",
                    "--weights",
                    "yolov5s.pt",
                ]
                subprocess.run(cmd, check=True, text=True, capture_output=True)
                pbar.update(1)
            return render_template(
                "train_yolo.html", message="Trening YOLOv5 zakończony sukcesem!"
            )
        except subprocess.CalledProcessError as e:
            return render_template(
                "train_yolo.html", message=f"Błąd przy trenowaniu YOLOv5:\n{e.stderr}"
            )
    else:
        return render_template("train_yolo.html", message=None)


# -----------------------------------------------------------
# TEST (val.py)
# -----------------------------------------------------------
@app.route("/test_yolo", methods=["GET", "POST"])
def test_yolo():
    console_output = ""
    if request.method == "POST":
        best_weights = os.path.join(
            "yolov5", "runs", "train", "exp3", "weights", "best.pt"
        )
        if not os.path.exists(best_weights):
            best_weights = "yolov5s.pt"
        cmd = [
            sys.executable,
            "yolov5/val.py",
            "--weights",
            best_weights,
            "--data",
            "gtsrb.yaml",
            "--task",
            "val",
            "--conf",
            "0.25",
        ]
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            out, err = process.communicate()
            if "AssertionError" in err and "trained on different --data" in err:
                console_output = (
                    "BŁĄD: Trenujesz/walidujesz różną liczbą klas. "
                    "Użyj wag zgodnych z GTSRB (43 klasy) lub zmień dane.\n\n"
                    + out
                    + "\n"
                    + err
                )
            else:
                console_output = out + "\n" + err
        except Exception as e:
            console_output = f"Błąd: {e}"
    return render_template("test_yolo.html", console_output=console_output)


# -----------------------------------------------------------
# TEST YOLO ON SINGLE IMAGE
# -----------------------------------------------------------
@app.route("/test_yolo_image", methods=["GET", "POST"])
def test_yolo_image():
    global yolo_model
    detected_image_path = None
    message = None
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template(
                "test_yolo_image.html",
                message="Nie wybrano obrazu.",
                detected_image_path=None,
            )
        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"], secure_filename(file.filename)
        )
        file.save(filepath)
        best_weights = os.path.join(
            "yolov5", "runs", "train", "exp3", "weights", "best.pt"
        )
        if not os.path.exists(best_weights):
            best_weights = "yolov5s.pt"
        yolo_model = torch.hub.load(
            "yolov5", "custom", path=best_weights, source="local", force_reload=True
        )
        results = yolo_model(filepath, size=640)
        results.render()
        rendered_image = results.ims[0]
        filename_only = os.path.basename(filepath)
        detected_name = "detected_" + filename_only
        detected_image_path = os.path.join(app.config["UPLOAD_FOLDER"], detected_name)
        cv2.imwrite(
            detected_image_path, cv2.cvtColor(rendered_image, cv2.COLOR_RGB2BGR)
        )
        message = "Wykonano detekcję!"
        return render_template(
            "test_yolo_image.html",
            message=message,
            detected_image_path=detected_image_path,
        )
    return render_template(
        "test_yolo_image.html", message=message, detected_image_path=detected_image_path
    )


# -----------------------------------------------------------
# TEST YOLO ON YOUTUBE (yt-dlp)
# -----------------------------------------------------------
@app.route("/test_yolo_youtube", methods=["GET", "POST"])
def test_yolo_youtube():
    global yolo_model
    message = None
    video_html_path = None
    if request.method == "POST":
        youtube_url = request.form.get("youtube_url")
        if not youtube_url:
            return render_template(
                "test_yolo_youtube.html",
                message="Proszę podać link.",
                video_html_path=None,
            )
        try:
            filename = "youtube_temp.mp4"
            local_video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            if os.path.exists(local_video_path):
                os.remove(local_video_path)
            ydl_opts = {
                "outtmpl": local_video_path,
                "format": "mp4",
                "nopart": True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
            best_weights = os.path.join(
                "yolov5", "runs", "train", "exp3", "weights", "best.pt"
            )
            if not os.path.exists(best_weights):
                best_weights = "yolov5s.pt"
            yolo_model = torch.hub.load(
                "yolov5", "custom", path=best_weights, source="local", force_reload=True
            )
            cap = cv2.VideoCapture(local_video_path)
            if not cap.isOpened():
                message = "Nie udało się otworzyć pobranego wideo."
                return render_template("test_yolo_youtube.html", message=message)
            out_filename = "detected_youtube.mp4"
            detected_video_path = os.path.join(
                app.config["UPLOAD_FOLDER"], out_filename
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            out_writer = cv2.VideoWriter(
                detected_video_path, fourcc, fps, (width, height)
            )
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                results = yolo_model(frame, size=640)
                results.render()
                rendered_frame = results.ims[0]
                rendered_frame_bgr = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)
                out_writer.write(rendered_frame_bgr)
            cap.release()
            out_writer.release()
            video_html_path = url_for("static", filename="uploads/" + out_filename)
            message = "Sukces! Wideo z YOLO zapisane."
        except Exception as e:
            message = f"Błąd przy pobieraniu/detekcji wideo z YouTube: {e}"
        return render_template(
            "test_yolo_youtube.html", message=message, video_html_path=video_html_path
        )
    else:
        return render_template(
            "test_yolo_youtube.html", message=None, video_html_path=None
        )


# -----------------------------------------------------------
# DETECT VIDEO (local file)
# -----------------------------------------------------------
@app.route("/video_yolo", methods=["GET", "POST"])
def video_yolo():
    global yolo_model, yolo_video_capture, yolo_video_active
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template(
                "video_yolo.html", message="Nie wybrano pliku wideo."
            )
        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"], secure_filename(file.filename)
        )
        file.save(filepath)
        best_weights = os.path.join(
            "yolov5", "runs", "train", "exp3", "weights", "best.pt"
        )
        if not os.path.exists(best_weights):
            best_weights = "yolov5s.pt"
        yolo_model = torch.hub.load(
            "yolov5", "custom", path=best_weights, source="local", force_reload=True
        )
        if yolo_video_capture:
            yolo_video_capture.release()
        yolo_video_capture = cv2.VideoCapture(filepath)
        yolo_video_active = True
        return render_template(
            "video_yolo.html",
            message="Wideo załadowane do YOLOv5. Kliknij 'Play Video'.",
        )
    else:
        return render_template("video_yolo.html")


def generate_video_frames_yolo():
    global yolo_video_capture, yolo_model, yolo_video_active
    while yolo_video_active and yolo_video_capture and yolo_video_capture.isOpened():
        ret, frame = yolo_video_capture.read()
        if not ret:
            break
        results = yolo_model(frame, size=640)
        boxes = results.xyxy[0]
        for *xyxy, conf, cls_id in boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            confidence = float(conf)
            class_id = int(cls_id)
            class_name = results.names.get(class_id, f"class_{class_id}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{class_name} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        ret2, buffer = cv2.imencode(".jpg", frame)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"


@app.route("/video_feed_yolo")
def video_feed_yolo():
    return Response(
        generate_video_frames_yolo(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# -----------------------------------------------------------
# (NEW) ADVANCED AUGMENTATION WITH ALBUMENTATIONS
# -----------------------------------------------------------
@app.route("/augment_data_advanced", methods=["GET", "POST"])
def augment_data_advanced():
    """
    We'll read from 'yolov5/my_gtsrb_yolo' (already converted to YOLO).
    For each image in 'train', apply a random set of Albumentations:
    We'll keep the validation set as-is (unaugmented).
    """
    if request.method == "POST":
        src_train_img_dir = "yolov5/my_gtsrb_yolo/images/train"
        src_train_lbl_dir = "yolov5/my_gtsrb_yolo/labels/train"
        src_val_img_dir = "yolov5/my_gtsrb_yolo/images/val"
        src_val_lbl_dir = "yolov5/my_gtsrb_yolo/labels/val"
        out_dir = "my_gtsrb_aug_advanced"
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(os.path.join(out_dir, "images/train"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "images/val"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels/train"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels/val"), exist_ok=True)
        # 1) Copy val images & labels as-is
        for fname in os.listdir(src_val_img_dir):
            shutil.copy(
                os.path.join(src_val_img_dir, fname),
                os.path.join(out_dir, "images/val", fname),
            )
        for fname in os.listdir(src_val_lbl_dir):
            shutil.copy(
                os.path.join(src_val_lbl_dir, fname),
                os.path.join(out_dir, "labels/val", fname),
            )
        transform = A.Compose(
            [
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.RGBShift(p=0.3),
                A.Rotate(limit=5, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                A.Perspective(scale=(0.02, 0.04), p=0.5),
                A.HorizontalFlip(p=0.5),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                check_each_transform=False,  # Wyłączenie automatycznego sprawdzania po każdej transformacji
                min_visibility=0.1,
            ),
        )
        import glob

        train_img_files = sorted(glob.glob(os.path.join(src_train_img_dir, "*.jpg")))
        for img_path in tqdm(
            train_img_files, desc="Wykonywanie zaawansowanej augmentacji"
        ):
            img_name = os.path.basename(img_path)
            name_no_ext, _ = os.path.splitext(img_name)
            label_path = os.path.join(src_train_lbl_dir, name_no_ext + ".txt")
            if not os.path.exists(label_path):
                continue
            image = cv2.imread(img_path)
            if image is None:
                continue
            h, w, _ = image.shape
            bboxes = []
            class_labels = []
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    items = line.strip().split()
                    if len(items) != 5:
                        continue
                    cls_id = int(items[0])
                    x_c = float(items[1])
                    y_c = float(items[2])
                    w_n = float(items[3])
                    h_n = float(items[4])
                    bboxes.append([x_c, y_c, w_n, h_n])
                    class_labels.append(cls_id)
            # Zapis oryginału
            orig_out_img_path = os.path.join(out_dir, "images/train", img_name)
            cv2.imwrite(orig_out_img_path, image)
            orig_out_lbl_path = os.path.join(
                out_dir, "labels/train", name_no_ext + ".txt"
            )
            with open(orig_out_lbl_path, "w") as fo:
                for cls_id, (x_c, y_c, ww, hh) in zip(class_labels, bboxes):
                    fo.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {ww:.6f} {hh:.6f}\n")
            # Augmentowana wersja
            # Po transformacji
            try:
                transformed = transform(
                    image=image, bboxes=bboxes, class_labels=class_labels
                )
            except ValueError:
                # Pomijamy tę klatkę lub zapisujemy obraz bez bboxes
                continue

            aug_image = transformed["image"]
            aug_bboxes = transformed["bboxes"]
            aug_labels = transformed["class_labels"]

            aug_img_name = name_no_ext + "_aug.jpg"
            aug_out_img_path = os.path.join(out_dir, "images/train", aug_img_name)
            cv2.imwrite(aug_out_img_path, aug_image)
            aug_lbl_name = name_no_ext + "_aug.txt"
            aug_out_lbl_path = os.path.join(out_dir, "labels/train", aug_lbl_name)
            with open(aug_out_lbl_path, "w") as fo2:
                for cls_id, (x_c, y_c, ww, hh) in zip(aug_labels, aug_bboxes):
                    # Ręczne przycięcie współrzędnych bboxów do zakresu [0,1]
                    x_c = max(0, min(x_c, 1))
                    y_c = max(0, min(y_c, 1))
                    ww = max(0, min(ww, 1))
                    hh = max(0, min(hh, 1))
                    fo2.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {ww:.6f} {hh:.6f}\n")
        adv_yaml = """train: my_gtsrb_aug_advanced/images/train
val: my_gtsrb_aug_advanced/images/val

names:
"""
        for i in range(len(sign_names)):
            adv_yaml += f"  - '{sign_names[str(i)]}'\n"
        with open("gtsrb_aug_advanced.yaml", "w") as fy:
            fy.write(adv_yaml)
        return render_template(
            "augment_data.html",
            message="Advanced augmentation completed! gtsrb_aug_advanced.yaml created.",
        )
    else:
        return render_template("augment_data.html", message=None)


# -----------------------------------------------------------
# (NEW) FINE-TUNE USING THE AUGMENTED DATASET
# -----------------------------------------------------------
@app.route("/fine_tune_advanced", methods=["GET", "POST"])
def fine_tune_advanced():
    if request.method == "POST":
        best_weights = os.path.join(
            "yolov5", "runs", "train", "exp3", "weights", "best.pt"
        )
        if not os.path.exists(best_weights):
            best_weights = "yolov5s.pt"
        try:
            with tqdm(
                total=1, desc="Fine-tuning YOLOv5 (advanced)", unit="krok"
            ) as pbar:
                cmd = [
                    sys.executable,
                    "yolov5/train.py",
                    "--img",
                    "640",
                    "--batch",
                    "16",
                    "--epochs",
                    "10",
                    "--data",
                    "gtsrb_aug_advanced.yaml",
                    "--weights",
                    best_weights,
                    "--project",
                    "yolov5/runs/train",
                    "--name",
                    "exp_finetune_advanced",
                ]
                subprocess.run(cmd, check=True, text=True, capture_output=True)
                pbar.update(1)
            return render_template(
                "fine_tune.html",
                message="Fine-tuning on advanced dataset zakończone sukcesem!",
            )
        except subprocess.CalledProcessError as e:
            return render_template(
                "fine_tune.html", message=f"Błąd przy fine-tuningu:\n{e.stderr}"
            )
    else:
        return render_template("fine_tune.html", message=None)


if __name__ == "__main__":
    app.run(debug=True)
