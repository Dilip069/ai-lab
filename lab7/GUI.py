import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, logging
from PIL import Image
import torch
from gtts import gTTS
from deep_translator import GoogleTranslator
from io import BytesIO
import threading

# Suppress Transformers warnings
logging.set_verbosity_error()

# Lock for thread-safe model use
model_lock = threading.Lock()


@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        local_files_only=False
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        local_files_only=False
    )

    return processor, model


# Load model
processor, model = load_model()

# Title
st.title("Image Captioning with Nepali Translation and Speech")

st.write("Purwanchal Campus, IOE")
st.write("Artificial Intelligence")

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# Camera input
camera_file = st.camera_input("Or capture from camera")

# Select image
image_file = uploaded_file or camera_file


if image_file:

    # Open image
    image = Image.open(image_file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Selected Image",
        use_container_width=True
    )

    st.subheader("Caption:")

    # Generate caption
    with model_lock:

        inputs = processor(
            image,
            return_tensors="pt"
        )

        with torch.no_grad():
            out = model.generate(**inputs)

        caption = processor.decode(
            out[0],
            skip_special_tokens=True
        )

    # English caption
    st.write("**English:**", caption)

    # Translate English → Nepali
    translated = GoogleTranslator(
        source="en",
        target="ne"
    ).translate(caption)

    st.write("**Nepali:**", translated)

    # Convert Nepali text to speech
    tts = gTTS(
        text=translated,
        lang="ne"
    )

    # Store MP3 in memory
    mp3_bytes = BytesIO()
    tts.write_to_fp(mp3_bytes)
    mp3_bytes.seek(0)

    # Play audio
    st.audio(
        mp3_bytes.read(),
        format="audio/mp3"
    )