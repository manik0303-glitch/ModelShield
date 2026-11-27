
import streamlit as st
import hashlib
import io
from PIL import Image

st.set_page_config(
    page_title="ModelShield - Red Team",
    page_icon="🛡️",
    layout="wide",
)

st.title("ModelShield")

from model.ram_adapter import (
    get_ram_model,
    preprocess_pil,
    get_baseline,
    new_investigation_id,
    MODEL_METADATA,
)

if st.session_state.get("ms_mission_mode") != "📜 Replay Audit":
    with st.sidebar:
        st.markdown("### 🔬 Active Investigation")

        uploaded = st.file_uploader(
            "Upload investigation image",
            type=["jpg", "jpeg", "png"],
            key="red_team_upload",
        )

        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            image_hash = hashlib.sha256(image_bytes).hexdigest()

            previous = st.session_state.get("investigation", {})

            if previous.get("_bytes_hash") != image_hash:
                st.session_state.pop("investigation", None)
                st.session_state.pop("current_mission_run", None)
                st.session_state.pop("feature_red_team", None)

                st.session_state["active_sample_idx"] = 0
                st.session_state["active_path_node"] = "DECISION"

                try:
                    image = Image.open(
                        io.BytesIO(image_bytes)
                    ).convert("RGB")

                    with st.spinner("Running RAM++ baseline..."):
                        model, device = get_ram_model()
                        pixel_tensor = preprocess_pil(image)

                        baseline = get_baseline(
                            pixel_tensor, model, device
                        )

                    st.session_state["investigation"] = {
                        "id": new_investigation_id(),
                        "_bytes_hash": image_hash,
                        "image_bytes": image_bytes,
                        "image_pil": image,
                        "pixel_tensor": pixel_tensor,
                        "filename": uploaded.name,
                        "ram_baseline": baseline,
                        "model_meta": MODEL_METADATA,
                    }

                except Exception as error:
                    st.error(f"RAM++ baseline failed: {error}")
                    st.stop()

            investigation = st.session_state["investigation"]

            st.image(
                investigation["image_pil"],
                caption=investigation["filename"],
                use_container_width=True,
            )

            st.success("RAM++ baseline ready")

            st.caption(
                f"Investigation: {investigation['id']}"
            )

            st.write("Recognized tags:")

            st.write(
                ", ".join(
                    investigation["ram_baseline"].get("tags", [])
                )
            )

        else:
            st.session_state.pop("investigation", None)
            st.session_state.pop("current_mission_run", None)
            st.info("Upload an image to begin.")

from orchestrator.ui import render_red_team_mission_lab

render_red_team_mission_lab()
