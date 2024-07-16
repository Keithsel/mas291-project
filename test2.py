import streamlit as st
import time

progress_text = "Operation in progress. Please wait."

with st.status("Downloading data...", expanded=True) as status:
    # create 2 progress bars, one for the whole operation and one for the current task
    pbar1 = st.progress(0)
    pbar2 = st.progress(0)
    # create a placeholder for the current task
    my_bar = st.empty()
    # update the progress bars and the placeholder
    for i in range(100):
        pbar1.progress(i)
        pbar2.progress(i)
        my_bar.text(progress_text + f" {i}%")
        time.sleep(0.1)
    # time.sleep(1)
    # my_bar.empty()
    status.update(label="Download complete!", state="complete", expanded=False)

st.button("Rerun")