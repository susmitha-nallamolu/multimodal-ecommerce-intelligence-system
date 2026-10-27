
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_DIR / "models" / "transformer_distilbert_3epoch"
CHECKPOINT_PATH = MODEL_DIR / "best_model.pth"

COMPARISON_PATH = PROJECT_DIR / "reports" / "model_comparison.csv"
CHART_PATH = PROJECT_DIR / "reports" / "model_comparison.png"

CATALOG_PATH = (
    PROJECT_DIR / "data" / "raw" / "processed" / "val_processed.csv"
)

IMAGE_DIR = PROJECT_DIR / "data" / "raw" / "train" / "train"

NUM_CLASSES = 21


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multimodal E-Commerce Intelligence",
    page_icon="🛍️",
    layout="wide",
)

st.title("🛍️ Multimodal E-Commerce Intelligence System")

st.markdown(
    """
    Explore products and use a trained Transformer model to predict
    product categories from titles and descriptions.
    """
)

st.info(
    "Current features: Transformer-based text prediction, "
    "model performance comparison, and product catalogue. "
    "Image-based AI prediction will be added separately."
)


# ============================================================
# 3. LOAD THE TRAINED TRANSFORMER
# ============================================================

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased",
        num_labels=NUM_CLASSES,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return tokenizer, model


# ============================================================
# 4. LOAD CATEGORY MAPPING
# ============================================================

@st.cache_data
def load_categories():
    mapping_path = (
        PROJECT_DIR
        / "data"
        / "raw"
        / "processed"
        / "label_mapping.csv"
    )

    return pd.read_csv(mapping_path)


# ============================================================
# 5. AI PRODUCT CATEGORY PREDICTION
# ============================================================

st.divider()
st.header("🤖 AI Product Category Prediction")

st.write(
    "Enter a product title and description. "
    "The trained DistilBERT Transformer will predict its category."
)

title_input = st.text_input(
    "Product title",
    placeholder="Example: Wireless Bluetooth Headphones",
    key="prediction_title",
)

description_input = st.text_area(
    "Product description",
    placeholder=(
        "Describe product features, specifications, "
        "and intended use..."
    ),
    height=130,
    key="prediction_description",
)

if st.button("Predict Product Category", type="primary"):

    product_text = (
        f"{title_input.strip()} {description_input.strip()}"
    ).strip()

    if not product_text:
        st.warning("Please enter a product title or description.")

    else:
        try:
            with st.spinner("Loading model and predicting category..."):

                tokenizer, model = load_model()
                categories = load_categories()

                inputs = tokenizer(
                    product_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128,
                )

                with torch.inference_mode():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(
                        outputs.logits,
                        dim=1,
                    )[0]

                predicted_id = int(
                    torch.argmax(probabilities).item()
                )

                confidence = float(
                    probabilities[predicted_id].item()
                )

                id_column = next(
                    (
                        column
                        for column in categories.columns
                        if column.lower()
                        in {
                            "label_id",
                            "label",
                            "id",
                            "class_id",
                        }
                    ),
                    None,
                )

                name_column = next(
                    (
                        column
                        for column in categories.columns
                        if column.lower()
                        in {
                            "category",
                            "category_name",
                            "name",
                            "label_name",
                        }
                    ),
                    None,
                )

                if id_column is None or name_column is None:
                    st.error(
                        "Could not identify the category ID and "
                        "category name columns in label_mapping.csv."
                    )

                else:
                    category_ids = pd.to_numeric(
                        categories[id_column],
                        errors="coerce",
                    )

                    match = categories[
                        category_ids == predicted_id
                    ]

                    if match.empty:
                        st.error(
                            f"No category name found for ID {predicted_id}."
                        )

                    else:
                        category_name = str(
                            match.iloc[0][name_column]
                        )

                        st.success("Prediction completed!")

                        col1, col2 = st.columns(2)

                        col1.metric(
                            "Predicted category",
                            category_name,
                        )

                        col2.metric(
                            "Model confidence",
                            f"{confidence:.2%}",
                        )

                        st.caption(
                            "Confidence is the model's estimated "
                            "probability, not a guarantee of correctness."
                        )

        except Exception as error:
            st.error(f"Prediction error: {error}")


# ============================================================
# 6. MODEL PERFORMANCE DASHBOARD
# ============================================================

st.divider()
st.header("📊 Model Performance Comparison")

if COMPARISON_PATH.exists():

    comparison = pd.read_csv(COMPARISON_PATH)

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    if CHART_PATH.exists():
        st.image(
            str(CHART_PATH),
            caption="Five-Model Product Classification Comparison",
            use_container_width=True,
        )

    else:
        st.info("The saved model comparison chart was not found.")

else:
    st.warning(
        "The saved model_comparison.csv report was not found."
    )


# ============================================================
# 7. PRODUCT CATALOGUE
# ============================================================

st.divider()
st.header("🛒 Product Catalogue")

st.write(
    "Browse sample products, search by title or description, "
    "and filter products by category."
)


@st.cache_data
def load_catalogue():
    if not CATALOG_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(CATALOG_PATH)


catalogue = load_catalogue()

if catalogue.empty:

    st.error(
        "The product dataset was not found or is empty.\n\n"
        f"Expected file: {CATALOG_PATH}"
    )

else:

    required_columns = {
        "ImgId",
        "title",
        "description",
        "categories",
    }

    missing_columns = required_columns - set(catalogue.columns)

    if missing_columns:

        st.error(
            "Missing catalogue columns: "
            + ", ".join(sorted(missing_columns))
        )

        st.write(
            "Available columns:",
            catalogue.columns.tolist(),
        )

    else:

        # ----------------------------------------------------
        # 7.1 SEARCH PRODUCTS
        # ----------------------------------------------------

        search_text = st.text_input(
            "🔍 Search products",
            placeholder="Example: wireless, headphones, phone...",
            key="catalogue_search",
        )

        # ----------------------------------------------------
        # 7.2 FILTER BY CATEGORY
        # ----------------------------------------------------

        category_options = sorted(
            catalogue["categories"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_category = st.selectbox(
            "Filter by category",
            ["All Categories"] + category_options,
            key="catalogue_category",
        )

        # ----------------------------------------------------
        # 7.3 APPLY FILTERS
        # ----------------------------------------------------

        filtered = catalogue.copy()

        if search_text.strip():

            query = search_text.strip()

            title_matches = (
                filtered["title"]
                .fillna("")
                .astype(str)
                .str.contains(
                    query,
                    case=False,
                    regex=False,
                )
            )

            description_matches = (
                filtered["description"]
                .fillna("")
                .astype(str)
                .str.contains(
                    query,
                    case=False,
                    regex=False,
                )
            )

            filtered = filtered[
                title_matches | description_matches
            ]

        if selected_category != "All Categories":

            filtered = filtered[
                filtered["categories"].astype(str)
                == selected_category
            ]

        st.write(
            f"**Matching products:** {len(filtered):,}"
        )

        # Display a maximum of 60 products per filtered view.
        products_to_display = filtered.head(60)

        if products_to_display.empty:

            st.info(
                "No matching products found. "
                "Try another search term or category."
            )

        else:

            # ------------------------------------------------
            # 7.4 DISPLAY TWO PRODUCT CARDS PER ROW
            # ------------------------------------------------

            cards_per_row = 2

            for start in range(
                0,
                len(products_to_display),
                cards_per_row,
            ):

                product_columns = st.columns(cards_per_row)

                row_products = products_to_display.iloc[
                    start:start + cards_per_row
                ]

                for column, (_, product) in zip(
                    product_columns,
                    row_products.iterrows(),
                ):

                    with column:

                        # Product image path
                        image_id = str(product["ImgId"])
                        # Prefer a high-resolution image for website display.
                        DISPLAY_IMAGE_DIR = PROJECT_DIR / "assets" / "catalogue_images"

                        high_res_path = DISPLAY_IMAGE_DIR / f"{image_id}.jpg"
                        original_path = IMAGE_DIR / f"{image_id}.jpg"

                        if high_res_path.exists():
                            image_path = high_res_path
                        else:
                            image_path = original_path

                        # ------------------------------------
                        # DISPLAY IMAGE
                        # ------------------------------------

                        # Display the original image without enlarging it.
                        if image_path.exists():
                            try:
                                with Image.open(image_path) as img:
                                    product_image = img.convert("RGB")
                                    st.image(
                                        product_image,
                                        width=100,
                                    )
                            except Exception:
                                st.warning("Could not display this product image.")
                        else:
                            st.info("Product image unavailable.")

                        # ------------------------------------
                        # PRODUCT TITLE
                        # ------------------------------------

                        product_title = str(product["title"])

                        st.markdown(
                            f"**{product_title}**"
                        )

                        # ------------------------------------
                        # PRODUCT DESCRIPTION
                        # ------------------------------------

                        description = product["description"]

                        if pd.isna(description):

                            description = (
                                "No description available."
                            )

                        else:

                            description = str(description)

                        if len(description) > 250:

                            description = (
                                description[:250] + "..."
                            )

                        st.write(description)

                        # ------------------------------------
                        # DATASET CATEGORY
                        # ------------------------------------

                        st.caption(
                            "Dataset category: "
                            + str(product["categories"])
                        )

                        st.divider()

            if len(filtered) > 60:

                st.info(
                    "Showing the first 60 matching products. "
                    "Use search or category filters to narrow "
                    "your results."
                )


# ============================================================
# 8. FOOTER
# ============================================================

st.divider()

st.caption(
    "Multimodal E-Commerce Intelligence System | "
    "AI-powered product classification demonstration"
)

