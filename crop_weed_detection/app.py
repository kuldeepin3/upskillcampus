import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Page Config
st.set_page_config(
    page_title="Crop & Weed Detection",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Crop & Weed Detection")
st.write("Upload an agricultural field image to detect crops and weeds.")

# Load Model
model = YOLO("Notebook/runs/detect/train-2/weights/best.pt")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Detect"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as tmp:

            image.save(tmp.name)

            results = model.predict(
                source=tmp.name,
                conf=0.25
            )

        result_image = results[0].plot()

        st.subheader("Detection Result")

        st.image(
            result_image,
            use_container_width=True
        )

        boxes = results[0].boxes

        st.subheader("Detected Objects")

        if len(boxes) == 0:
            st.warning("No Crop or Weed detected.")
        else:

            names = model.names

            for box in boxes:

                cls = int(box.cls)

                conf = float(box.conf)

                st.write(
                    f"**{names[cls]}** : {conf:.2f}"
                )