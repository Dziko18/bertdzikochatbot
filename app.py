import streamlit as st
import pandas as pd
import re

from pathlib import Path
from sentence_transformers import SentenceTransformer, util


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SehatQ Medical Chatbot",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ==========================================
# CUSTOM UI
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #F5F7FC;
}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #253C68;
}

[data-testid="stSidebar"] {
    background-color: #EAF0FA;
}

[data-testid="stChatMessage"] {
    border: 1px solid #E1E8F4;
    border-radius: 14px;
    padding: 14px;
}

div.stButton > button {
    border-radius: 10px;
}

[data-testid="stChatInput"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# TEXT NORMALIZATION
# ==========================================

def normalize_text(text):

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================
# LOAD DATASET AND MODEL
# ==========================================

@st.cache_resource(show_spinner=False)
def load_chatbot():

    # Lokasi dataset asli
    file_path = (
        Path(__file__).resolve().parent
        / "doctor_id_qa_original.csv"
    )

    # Pastikan file tersedia
    if not file_path.exists():

        raise FileNotFoundError(
            "Dataset doctor_id_qa_original.csv "
            "tidak ditemukan di repository."
        )

    # Load CSV
    df = pd.read_csv(file_path)

    # Normalisasi nama kolom
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # Validasi kolom
    required_columns = {
        "question",
        "answer",
        "split"
    }

    if not required_columns.issubset(df.columns):

        raise ValueError(
            f"Kolom dataset tidak sesuai: "
            f"{df.columns.tolist()}"
        )

    # ======================================
    # DATA CLEANING
    # ======================================

    # Hapus missing values
    df = df.dropna(
        subset=[
            "question",
            "answer",
            "split"
        ]
    ).copy()

    # Normalisasi pertanyaan dan jawaban
    df["question"] = (
        df["question"]
        .apply(normalize_text)
    )

    df["answer"] = (
        df["answer"]
        .apply(normalize_text)
    )

    # Normalisasi split
    df["split"] = (
        df["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Gunakan data train saja
    df = df[
        df["split"] == "train"
    ].copy()

    # Hapus teks kosong
    df = df[
        (df["question"] != "") &
        (df["answer"] != "")
    ]

    # Hapus duplikasi Q&A
    df = df.drop_duplicates(
        subset=[
            "question",
            "answer",
            "split"
        ]
    )

    # Hapus pertanyaan duplikat
    df["question_key"] = (
        df["question"].str.casefold()
    )

    df = df.drop_duplicates(
        subset=["question_key"]
    )

    df = df.reset_index(drop=True)

    if df.empty:

        raise ValueError(
            "Dataset kosong setelah cleaning."
        )

    # ======================================
    # PREPARE KNOWLEDGE BASE
    # ======================================

    questions = (
        df["question"]
        .astype(str)
        .tolist()
    )

    answers = (
        df["answer"]
        .astype(str)
        .tolist()
    )

    # ======================================
    # LOAD MODEL E5
    # ======================================

    model = SentenceTransformer(
        "intfloat/multilingual-e5-base",
        device="cpu"
    )

    # Prefix passage untuk E5
    passages = [
        "passage: " + question
        for question in questions
    ]

    # Generate embedding
    question_embeddings = model.encode(
        passages,
        convert_to_tensor=True,
        normalize_embeddings=True,
        batch_size=8,
        show_progress_bar=False
    )

    return (
        df,
        questions,
        answers,
        model,
        question_embeddings
    )


# ==========================================
# INITIALIZE CHATBOT
# ==========================================

with st.spinner(
    "Menyiapkan SehatQ Medical Chatbot..."
):

    (
        df,
        questions,
        answers,
        model,
        question_embeddings
    ) = load_chatbot()


# ==========================================
# MEDICAL CHATBOT FUNCTION
# ==========================================

def medical_chatbot(user_question):

    user_question = normalize_text(
        user_question
    )

    if not user_question:

        return (
            "Silakan masukkan pertanyaan.",
            "",
            0.0
        )

    # Prefix query untuk model E5
    query_embedding = model.encode(
        "query: " + user_question,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    # Semantic Search
    results = util.semantic_search(
        query_embedding,
        question_embeddings,
        top_k=1
    )[0]

    # Ambil hasil terdekat
    best_match = results[0]

    index = int(
        best_match["corpus_id"]
    )

    score = float(
        best_match["score"]
    )

    # Ambil jawaban dari dataset
    answer = answers[index]

    matched_question = questions[index]

    return (
        answer,
        matched_question,
        score
    )


# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🩺 SehatQ")

    st.caption(
        "Medical Question Answering System"
    )

    st.divider()

    st.subheader("Informasi Dataset")

    st.metric(
        "Total Data",
        f"{len(df):,}"
    )

    st.write(
        "Dataset: Doctor QA Indonesia"
    )

    st.write(
        "Model: multilingual-e5-base"
    )

    st.write(
        "Metode: Semantic Search"
    )

    st.write(
        "Bahasa: Indonesia"
    )

    st.divider()

    st.subheader("Tentang SehatQ")

    st.write(
        "SehatQ merupakan chatbot berbasis "
        "Natural Language Processing yang "
        "menggunakan Sentence Transformers "
        "untuk mencari jawaban paling relevan "
        "dari dataset konsultasi kesehatan "
        "berbahasa Indonesia."
    )

    st.divider()

    st.subheader("Percakapan")

    if st.button(
        "Hapus Percakapan",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ==========================================
# MAIN HEADER
# ==========================================

st.title("🩺 SehatQ")

st.markdown(
    "Asisten Informasi Kesehatan"
)

st.caption(
    "Tanyakan informasi seputar penyakit, "
    "gejala, penyebab, dan penanganan umum."
)

st.divider()


# ==========================================
# WELCOME MESSAGE
# ==========================================

if not st.session_state.messages:

    st.info(
        "Halo! Ada yang ingin ditanyakan? "
        "Silakan ketik pertanyaan kesehatan "
        "pada kolom di bawah."
    )


# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message["role"] == "assistant":

            if "matched" in message:

                with st.expander(
                    "Lihat Detail Jawaban"
                ):

                    st.write(
                        "Pertanyaan Terdekat:"
                    )

                    st.write(
                        message["matched"]
                    )

                    st.write(
                        "Similarity Score:"
                    )

                    st.write(
                        f'{message["score"]:.4f}'
                    )


# ==========================================
# USER INPUT
# ==========================================

user_input = st.chat_input(
    "Tulis pertanyaan kesehatan..."
)


# ==========================================
# GENERATE CHATBOT RESPONSE
# ==========================================

if user_input:

    # Simpan pertanyaan user
    st.session_state.messages.append({

        "role": "user",

        "content": user_input

    })

    # Tampilkan pertanyaan
    with st.chat_message("user"):

        st.markdown(
            user_input
        )

    # Jalankan chatbot
    with st.chat_message("assistant"):

        with st.spinner(
            "SehatQ sedang mencari jawaban..."
        ):

            (
                answer,
                matched_question,
                score
            ) = medical_chatbot(
                user_input
            )

        # Tampilkan jawaban
        st.markdown(
            answer
        )

        # Detail hasil retrieval
        with st.expander(
            "Lihat Detail Jawaban"
        ):

            st.write(
                "Pertanyaan Terdekat:"
            )

            st.write(
                matched_question
            )

            st.write(
                "Similarity Score:"
            )

            st.write(
                f"{score:.4f}"
            )

    # Simpan jawaban chatbot
    st.session_state.messages.append({

        "role": "assistant",

        "content": answer,

        "matched": matched_question,

        "score": score

    })


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Powered by Doctor QA Indonesia Dataset "
    "| Multilingual E5 | Semantic Search"
)

st.caption(
    "Informasi kesehatan untuk tujuan edukasi. "
    "Bukan pengganti konsultasi dokter."
)
