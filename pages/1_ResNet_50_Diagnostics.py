import json
from io import BytesIO
from pathlib import Path

import streamlit as st

from evaluation.resnet_objects import DEFAULT_CHECKPOINT, diagnose


st.title("ResNet-50 gradient-masking diagnostics")
st.write("Upload up to eight photos and select the actual object in each. "
         "The detector uses the labels to measure which clean predictions were correct.")
checkpoint = st.text_input("Seven-class checkpoint", value=str(DEFAULT_CHECKPOINT))
photos = st.file_uploader("Labeled photos", type=["jpg", "jpeg", "png", "webp"],
                         accept_multiple_files=True)
examples = []
for index, photo in enumerate(photos[:8]):
    label = st.selectbox(f"Actual object in {photo.name}",
                         ["cat", "dog", "flower", "phone", "truck", "umbrella", "watch"],
                         index=None, key=f"object-{index}-{photo.file_id}")
    examples.append((photo, label))
if len(photos) > 8:
    st.warning("Select at most eight photos.")
if st.button("Run diagnostics", disabled=not examples or len(photos) > 8 or
             any(label is None for _, label in examples)):
    try:
        with st.spinner("Evaluating gradient-masking signals…"):
            result = diagnose(Path(checkpoint), [(BytesIO(photo.getvalue()), label)
                                                 for photo, label in examples])
        st.subheader(result["status"])
        st.write(result["summary"])
        st.json(result)
        st.download_button("Download report", json.dumps(result, indent=2), "resnet50-diagnostics.json")
    except (OSError, RuntimeError, KeyError, ValueError) as exc:
        st.error(str(exc))
