from ultralytics import YOLO
import cv2
import streamlit as st
import requests
import time
from PIL import Image
import numpy as np

# Load YOLO model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

API_URL = "http://127.0.0.1:8000/recommend"

st.title("🍳 Vietnomi")

# Chọn nguồn ảnh
mode = st.radio("Chọn nguồn đầu vào:", ["Camera", "Upload ảnh"])

FRAME_WINDOW = st.empty()
detected_box = st.empty()
recipe_box = st.empty()

# Session state để lưu ingredient + thời gian gọi API
if "detected_ingredients" not in st.session_state:
    st.session_state.detected_ingredients = set()
if "last_api_call" not in st.session_state:
    st.session_state.last_api_call = 0

API_DELAY = 3  # giây

INGREDIENT_MAP = {
    "beef": "thịt bò",
    "bell_pepper": "ớt chuông",
    "cabbage": "bắp cải",
    "carrot": "cà rốt",
    "cauliflower": "súp lơ",
    "chicken": "thịt gà",
    "cucumber": "dưa leo",
    "egg": "trứng",
    "fish": "cá",
    "garlic": "tỏi",
    "ginger": "gừng",
    "kumquat": "quất",
    "lemon": "chanh",
    "onion": "hành tây",
    "pork": "thịt heo",
    "potato": "khoai tây",
    "shrimp": "tôm",
    "small_pepper": "ớt",
    "tofu": "đậu phụ",
    "tomato": "cà chua"
}

# -------- Camera mode --------
if mode == "Camera":
    run = st.checkbox("Bật camera", value=False)
    cap = cv2.VideoCapture(0)

    if run:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Không mở được camera")
                break

            # YOLO detect
            results = model(frame, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf >= 0.7:
                        class_name_en = model.names[cls]
                        class_name_vi = INGREDIENT_MAP.get(class_name_en, class_name_en)
                        st.session_state.detected_ingredients.add(class_name_vi)

            # Render ảnh
            annotated_frame = results[0].plot()
            FRAME_WINDOW.image(annotated_frame, channels="BGR")

            # Show ingredient list
            detected_box.subheader("🥦 Nguyên liệu nhận diện (conf > 0.7):")
            detected_box.write(list(st.session_state.detected_ingredients))

            # Delay call API
            if (
                st.session_state.detected_ingredients
                and (time.time() - st.session_state.last_api_call > API_DELAY)
            ):
                try:
                    payload = {
                        "user_id": 0,
                        "detected_ingredients": list(st.session_state.detected_ingredients),
                        "basis": "content",
                        "k": 5,
                    }
                    response = requests.post(API_URL, json=payload, timeout=5)
                    if response.status_code == 200:
                        recipe_box.subheader("🍲 Gợi ý món ăn:")
                        recipe_box.json(response.json())
                    else:
                        recipe_box.error("Lỗi API backend")
                    st.session_state.last_api_call = time.time()
                except Exception as e:
                    recipe_box.error(f"Lỗi khi gọi API: {e}")

            time.sleep(0.05)
    else:
        cap.release()

# -------- Upload mode --------
elif mode == "Upload ảnh":
    img_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"])
    if img_file is not None:
        # Convert PIL → numpy (BGR cho YOLO giống camera)
        pil_img = Image.open(img_file).convert("RGB")
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # YOLO detect
        results = model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf >= 0.7:
                    class_name_en = model.names[cls]
                    class_name_vi = INGREDIENT_MAP.get(class_name_en, class_name_en)
                    st.session_state.detected_ingredients.add(class_name_vi)

        # Render ảnh
        annotated_frame = results[0].plot()
        st.image(annotated_frame, channels="BGR", caption="Kết quả detect")

        # Show ingredient list
        detected_box.subheader("🥦 Nguyên liệu nhận diện (conf > 0.7):")
        detected_box.write(list(st.session_state.detected_ingredients))

        # Gọi API (không cần loop như camera)
        if st.session_state.detected_ingredients:
            try:
                payload = {
                    "user_id": 0,
                    "detected_ingredients": list(st.session_state.detected_ingredients),
                    "basis": "content",
                    "k": 5,
                }
                response = requests.post(API_URL, json=payload, timeout=5)
                if response.status_code == 200:
                    recipe_box.subheader("🍲 Gợi ý món ăn:")
                    recipe_box.json(response.json())
                else:
                    recipe_box.error("Lỗi API backend")
            except Exception as e:
                recipe_box.error(f"Lỗi khi gọi API: {e}")
