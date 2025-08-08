import streamlit as st
from query_engine import query, sources as all_sources, generate_answer
import os
import ingest
import re
from evaluation_data import evaluation_sets

DATA_DIR = "data"

st.title("📄 LLM-Powered Document Search")
st.write("Upload PDFs or DOCX files and ask questions about their content.")

# --- Upload documents ---
uploaded_files = st.file_uploader(
    "📤 Upload your documents here",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        save_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"✅ Uploaded {len(uploaded_files)} file(s). Re-ingesting...")

    ingest.ingest()
    st.success("✅ Documents re-indexed successfully!")

st.markdown("---")

# ✅ Evaluation Mode
eval_mode = st.checkbox("🧪 Run in Evaluation Mode")

if eval_mode:
    st.subheader("📊 Evaluation Results")

    total = 0
    correct = 0

    for filename, qa_pairs in evaluation_sets.items():
        st.markdown(f"### 📁 {filename}")
        for pair in qa_pairs:
            question = pair["question"]
            expected = pair["answer"].lower()

            results = query(question)  # now a dict with answer + sources
            texts = " ".join([r['text'].lower() for r in results["sources"]])

            match = expected in texts
            total += 1
            correct += int(match)

            st.markdown(f"**Q:** {question}")
            st.markdown(f"**✅ Expected:** {pair['answer']}")
            st.markdown(f"**📄 Found in Results:** {'✅ Yes' if match else '❌ No'}")
            st.markdown("---")

    if total > 0:
        acc = round((correct / total) * 100, 2)
        st.success(f"🎯 Accuracy: {acc}% ({correct}/{total} correct)")
    else:
        st.warning("⚠️ No evaluation data matched uploaded files.")
    
    st.stop()

# --- Initialize chat history ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- File source filter ---
if all_sources:
    unique_sources = sorted(set(all_sources))
    selected_files = st.multiselect("📂 Filter by file(s)", unique_sources, default=unique_sources)
else:
    selected_files = []

# --- Highlight helper ---
def highlight(text, query):
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return pattern.sub(f"**{query}**", text)

# --- User query input ---
user_query = st.text_input("🔍 Enter your question:")

if user_query:
    results = query(user_query)  # dict: {answer, sources}
    filtered_results = [res for res in results["sources"] if res["source"] in selected_files]

    # If you still want to use generate_answer separately
    answer = generate_answer(user_query, filtered_results)

    st.session_state.history.append((user_query, filtered_results, answer))

    # --- Show AI answer ---
    st.subheader("💡 AI Answer")
    st.write(answer)

    # --- Show retrieved chunks ---
    st.subheader("📌 Top Retrieved Passages")
    for res in filtered_results:
        st.markdown(f"**Source:** {res['source']}")
        st.markdown(highlight(res['text'], user_query))
        st.markdown("---")

# --- Show query history ---
if st.session_state.history:
    st.markdown("### 💬 Previous Queries")
    for prev_query, prev_results, prev_answer in reversed(st.session_state.history):
        st.markdown(f"**Q: {prev_query}**")
        st.markdown(f"**💡 AI Answer:** {prev_answer}")
        for res in prev_results:
            st.markdown(f"- 📄 *{res['source']}*")
            st.markdown(highlight(res["text"], prev_query))
        st.markdown("------")
