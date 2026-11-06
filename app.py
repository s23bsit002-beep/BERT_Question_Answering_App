
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Hugging Face Model Name
MODEL_NAME = "khadija123it/BERT-Multilingual-QA"

# Page Configuration
st.set_page_config(
    page_title="BERT Question Answering System",
    page_icon="🤖",
    layout="wide"
)

# Application Heading
st.title("🤖 BERT Question Answering System")

st.write(
    "Enter a context and ask a question. "
    "Our trained BERT model will find the answer."
)

st.divider()

# Load Model and Tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

try:
    with st.spinner("Loading BERT Model..."):
        tokenizer, model = load_model()

    st.success("BERT Model Loaded Successfully!")

except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# Context Input
st.subheader("1. Enter Context")

context = st.text_area(
    "Context:",
    placeholder="Enter your paragraph here...",
    height=200
)

# Question Input
st.subheader("2. Enter Your Question")

question = st.text_input(
    "Question:",
    placeholder="Type your question here..."
)

# Answer Prediction
if st.button("Get Answer", type="primary"):

    if not context.strip() or not question.strip():

        st.warning("Please enter both context and question.")

    else:

        inputs = tokenizer(
            question,
            context,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation="only_second",
            max_length=384
        )

        offsets = inputs.pop("offset_mapping")[0].tolist()
        sequence_ids = inputs.sequence_ids()

        with torch.no_grad():
            outputs = model(**inputs)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # Select only context tokens
        valid_positions = [
            i for i, sid in enumerate(sequence_ids)
            if sid == 1 and offsets[i][1] > offsets[i][0]
        ]

        best_score = float("-inf")
        best_start = None
        best_end = None

        # Find the best answer span
        for start in valid_positions:
            for end in valid_positions:

                if end < start:
                    continue

                if end - start > 30:
                    break

                score = (
                    start_logits[start].item()
                    + end_logits[end].item()
                )

                if score > best_score:
                    best_score = score
                    best_start = start
                    best_end = end

        # Display Answer
        if best_start is not None:

            char_start = offsets[best_start][0]
            char_end = offsets[best_end][1]

            answer = context[char_start:char_end]

            st.divider()
            st.subheader("3. Predicted Answer")

            st.success(answer)

            st.write("Raw Answer Score:", round(best_score, 4))

        else:
            st.warning("No answer could be extracted.")

# Footer
st.divider()
st.caption("BERT Question Answering System | NLP Project")
