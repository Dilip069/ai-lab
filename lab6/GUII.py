import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import timm
import torch.nn.functional as F
from pathlib import Path

# Streamlit Page Configuration

st.set_page_config(
    page_title="Image Classifier",
    page_icon="🇳🇵",
    layout="wide",
)



# Device


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Load Trained Model

@st.cache_resource
def load_model():

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=10
    )

    # Find best_model.pth in the same folder as this Python file
    MODEL_PATH = Path(__file__).parent / "best_model.pth"

    # Load trained weights
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    )

    return model


model = load_model()

# Move model to device
model.to(device)
model.eval()



# Class Labels

idx_to_class = {
    0: "tench",
    1: "English springer",
    2: "cassette player",
    3: "chain saw",
    4: "church",
    5: "French horn",
    6: "garbage truck",
    7: "gas pump",
    8: "golf ball",
    9: "parachute"
}



# Image Transform


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])



# Prediction Function


def predict_image(image):

    # Preprocess image
    image = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():

        outputs = model(image)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        _, predicted = torch.max(
            outputs,
            1
        )

    return predicted.item(), probabilities.squeeze()



# Streamlit GUI


st.title("Imagenette Image Classifier using EfficientNet-B0")

st.write(
    "Upload an image from the Imagenette dataset, "
    "and the model will predict the class."
)



# Three Columns


col1, col2, col3 = st.columns([1, 1, 1])



# Column 1: Upload Image


with col1:

    st.header("Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "png", "jpeg"]
    )



# Column 2: Preview


if uploaded_file is not None:

    image = Image.open(uploaded_file)

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    with col2:

        st.header("Preview")

        resized_image = image.resize(
            (400, 300)
        )

        st.image(
            resized_image,
            caption="Uploaded Image",
            use_container_width=False
        )

        predict_clicked = st.button(
            "Predict"
        )



    # Column 3: Prediction


    with col3:

        st.header("Prediction")

        if predict_clicked:

            preds, probs = predict_image(image)

            predicted_class_name = idx_to_class[preds]

            st.markdown(
                f"""
                <h3 style='color: #4CAF50;'>
                Predicted: {predicted_class_name}
                </h3>
                """,
                unsafe_allow_html=True
            )

            # Confidence chart
            fig, ax = plt.subplots()

            ax.bar(
                list(idx_to_class.values()),
                probs.tolist(),
                color="skyblue"
            )

            ax.set_xlabel("Classes")
            ax.set_ylabel("Confidence")

            plt.xticks(
                rotation=45,
                ha="right"
            )

            plt.tight_layout()

            st.pyplot(fig)