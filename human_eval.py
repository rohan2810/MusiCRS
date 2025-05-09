import streamlit as st
import json, re
from pathlib import Path
from typing import List


@st.cache_data
def load_data(path: str) -> List[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def extract_urls(text: str) -> List[str]:
    raw = re.findall(r'(?:https?://|ftp://|www\.)\S+', text)
    return [u.rstrip(".,);]'\"") for u in raw]

def editable_entities(title: str, existing: List[str], key: str) -> List[str]:
    st.subheader(title)
    kept = st.multiselect(f"Keep these", options=existing, default=existing, key=f"{key}_keep")
    added_txt = st.text_input(f"Add more (comma-separated)", key=f"{key}_add")
    added = [x.strip() for x in added_txt.split(",") if x.strip()]
    return kept + added

def main():
    st.sidebar.title("Annotator Settings")
    annotator_id = st.sidebar.text_input("Annotator ID", help="Your name or unique ID")
    input_path   = st.sidebar.text_input("Input JSONL file", value="slice.jsonl")
    output_path  = st.sidebar.text_input("Output JSONL file", value="slice_annotations.jsonl")
    
    # allow file upload as alternative
    uploaded = st.sidebar.file_uploader("Or upload JSONL", type="jsonl")
    if uploaded is not None:
        data = [json.loads(line) for line in uploaded.getvalue().decode().splitlines()]
    else:
        data = load_data(input_path)
    total = len(data)
    
    st.sidebar.markdown(f"Total samples: **{total}**")
    start = st.sidebar.number_input("Start index", min_value=0, max_value=total-1, value=0)
    end   = st.sidebar.number_input("End index",   min_value=1, max_value=total,   value=total)
    if 'idx' not in st.session_state:
        st.session_state.idx = start
    st.session_state.idx = min(max(st.session_state.idx, start), end-1)
    i = st.session_state.idx
    sample = data[i]
    
    st.header(f"Sample {i+1} of {end-start} — Annotator: {annotator_id or '...' }")
    
    # 1) Editable query
    edited_query = st.text_area(
        "Query (editable)",
        value=sample.get("query", ""),
        height=120,
        key=f"query_{i}"
    )
    
    # 2) Show combined text
    st.subheader("Combined Text")
    st.markdown(sample.get("combined_text", "—"))
    
    # 3) URL extraction & keep checkboxes
    st.subheader("Extracted URLs")
    urls = extract_urls(sample.get("combined_text",""))
    kept_urls = []
    for url in urls:
        keep = st.checkbox(f"{url}", value=True, key=f"url_{i}_{url}")
        if keep:
            st.write(f"[{url}]({url})")
            kept_urls.append(url)
    
    # 4) Bucket selection (fixed mapping)
    bucket_map = {
        "music":         "Music post",
        "pop-culture":   "Popular culture post",
        "pop culture":   "Popular culture post",
        "other":         "Other post"
    }

    raw_bucket = sample.get("bucket", "").lower().replace("_", " ")
    gpt_suggestion = bucket_map.get(raw_bucket, "Other post")
    st.markdown(f"**GPT suggestion:** {gpt_suggestion}")

    default_bucket = bucket_map.get(raw_bucket, "Other post")
    bucket = st.radio(
        "Assign this post to a bucket",
        ["Music post", "Popular culture post", "Other post"],
        index=["Music post","Popular culture post","Other post"].index(default_bucket),
        key=f"bucket_{i}"
    )
    
    # 5) Entities: submission + comments, with manual add
    se = editable_entities(
        "Submission entities (songs)",
        sample.get("submission_entities", {}).get("songs", []),
        key=f"se_{i}"
    )
    cce = editable_entities(
        "Combined comment entities (songs)",
        sample.get("combined_comment_entities", {}).get("songs", []),
        key=f"cce_{i}"
    )
    
    # 6) Store annotation in session_state
    if "annotations" not in st.session_state:
        st.session_state.annotations = {}
    st.session_state.annotations[i] = {
        "annotator_id": annotator_id,
        "query": edited_query,
        "bucket": bucket,
        "submission_entities": se,
        "combined_comment_entities": cce,
        "valid_urls": kept_urls
    }
    
    # 7) Navigation
    def go_prev():
        st.session_state.idx = max(st.session_state.idx - 1, start)
    def go_next():
        st.session_state.idx = min(st.session_state.idx + 1, end-1)
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        st.button("← Previous", on_click=go_prev, disabled=(i <= start))
    with col3:
        st.button("Next →",     on_click=go_next, disabled=(i >= end-1))
    
    # 8) Save & download
    if st.button("Save annotations to file"):
        out = Path(output_path)
        with open(out, 'w', encoding='utf-8') as f:
            for idx, ann in sorted(st.session_state.annotations.items()):
                record = data[idx].copy()
                record.update(ann)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        st.success(f"Saved to {out}")
    
    if st.session_state.annotations:
        blob = "\n".join(
            json.dumps(data[idx] | ann, ensure_ascii=False)
            for idx, ann in sorted(st.session_state.annotations.items())
        )
        st.download_button(
            "Download annotations JSONL",
            data=blob,
            file_name=Path(output_path).name,
            mime="text/plain"
        )

if __name__ == "__main__":
    main()