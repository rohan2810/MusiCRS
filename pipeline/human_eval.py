import streamlit as st
import json, re, argparse
from pathlib import Path

@st.cache_data
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def extract_urls(text):
    # grab anything that starts with http(s)://, ftp:// or www.
    raw = re.findall(r'(?:https?://|ftp://|www\.)\S+', text)
    # strip trailing . , ) ; ] ' " characters
    return [u.rstrip(".,);]'\"") for u in raw]




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  "-i", required=True, help="Input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file for annotations")
    return parser.parse_args()

def main():
    args = parse_args()
    output_filename = st.sidebar.text_input(
        "Output JSONL filename",
        value=args.output,
        help="Enter the filename (or path) where you'd like to save annotations",
        key="output_filename"
    )    
    data = load_data(args.input)
    total = len(data)

    # Sidebar: slice
    st.sidebar.title("Data slice")
    start = st.sidebar.number_input("Start index", min_value=0, max_value=total-1, value=0)
    end   = st.sidebar.number_input("End index",   min_value=1, max_value=total,   value=total)

    # Clamp current pointer
    if 'idx' not in st.session_state:
        st.session_state.idx = start
    st.session_state.idx = min(max(st.session_state.idx, start), end-1)
    i = st.session_state.idx
    sample = data[i]

    st.header(f"Sample {i+1} of {end - start}")
    edited_query = st.text_area(
        "Query (you can edit)",
        value=sample.get("query", ""),
        height=150,
        key=f"query_{i}"
    )
    st.markdown(f"**Combined text:**  \n{sample['combined_text']}")
    st.write(f"**Original URL:** {sample.get('url','—')}")

    gpt_bucket = sample.get("bucket", "").lower()
    bucket_map = {
        "music":         "Music post",
        "pop-culture":   "Popular culture post",
        "other":         "Other post"
    }
    default_label = bucket_map.get(gpt_bucket, "Other post")

    st.markdown(f"**GPT suggestion:** {bucket_map.get(gpt_bucket, 'None')}")

    bucket = st.radio(
        "Assign this post to a bucket",
        ["Music post", "Popular culture post", "Other post"],
        index=["Music post", "Popular culture post", "Other post"].index(default_label),
        key=f"bucket_{i}"
    )
    # ------------------------------------------

    # 2) Entities labeling
    def entity_multiselect(title, entities, key):
        st.subheader(title)
        out = {}
        for ent_type in ["artists","albums","songs"]:
            opts = entities.get(ent_type, [])
            sel  = st.multiselect(
                f"{ent_type.title()}",
                options=opts,
                default=opts,
                key=f"{key}_{ent_type}"
            )
            out[ent_type] = sel
        return out

    se_sel  = entity_multiselect("Submission entities",        sample.get("submission_entities",{}),       f"se_{i}")
    cce_sel = entity_multiselect("Comment entities",           sample.get("combined_comment_entities",{}),  f"cce_{i}")

    st.subheader("URLs in combined_text (click to visit, then check to keep)")
    urls = extract_urls(sample.get("combined_text",""))
    valid_urls = []
    for url in urls:
        col1, col2 = st.columns([1, 9])
        keep = col1.checkbox("", value=True, key=f"url_{i}_{url}")
        # render as a clickable link
        col2.markdown(f"[{url}]({url})")
        if keep:
            valid_urls.append(url)


    # 4) Save into session_state
    st.session_state.annotations = st.session_state.get("annotations", {})
    st.session_state.annotations[i] = {
         "query": edited_query,   
        "bucket": bucket,
        "submission_entities": se_sel,
        "combined_comment_entities": cce_sel,
        "valid_urls": valid_urls
    }

    # 5) Navigation

    def go_prev():
        st.session_state.idx -= 1

    def go_next():
        st.session_state.idx += 1

    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        st.button(
            "← Previous",
            key=f"prev_{i}",
            on_click=go_prev,
            disabled=(i <= start)
        )
    with col3:
        st.button(
            "Next →",
            key=f"next_{i}",
            on_click=go_next,
            disabled=(i >= end-1)
        )


    # 6) Save & Download

    if st.button("Save all annotations to file"):
        out_path = Path(st.session_state.output_filename)
        # Overwrite each time, writing exactly one line per annotated index
        with open(out_path, 'w', encoding='utf-8') as f:
            for idx in sorted(st.session_state.annotations):
                rec = data[idx].copy()
                # merge in the full per-sample annotation dict (including query)
                rec.update(st.session_state.annotations[idx])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        st.success(f"Annotations saved to {out_path} (overwritten)")
    # offer download
    if "annotations" in st.session_state:
        annotated_jsonl = "\n".join(
            json.dumps(data[idx] | st.session_state.annotations.get(idx, {}), ensure_ascii=False)
            for idx in range(start, end)
        )
        st.download_button(
            "Download annotated JSONL",
            annotated_jsonl,
            file_name=Path(st.session_state.output_filename).name,
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
