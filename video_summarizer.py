from dotenv import load_dotenv
import google.generativeai as genai
from phi.agent import Agent
import streamlit as st
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file,get_file

import time
from pathlib import Path
import tempfile
import os

load_dotenv()

API_KEY=os.getenv("GEMINI_API_KEY")
if API_KEY :
    genai.configure(api_key=API_KEY)
    
    
st.set_page_config(
    page_title="Multimodal AI Agent- Video Summarizer",
    page_icon="🎥",
    layout="wide"
)
st.title("Phidata Video AI Summarizer Agent 🎥🎤🖬")
st.header("Powered by Gemini 2.0 Flash Exp")

def initialize_agent():
    return Agent(
        Name="Phidata Video AI Summarizer Agent",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
        
    )
    
## Initializing the agent
multimodal_Agent=initialize_agent()

video_file=st.file_uploader(
    "Upload the video",type=['mp4','mov','avi'],help="upload a video for the analysis"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False,suffix='.mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path=temp_video.name
        st.text(f"Uploading video: {video_path}")
        
        
    st.video(video_path,format="video/mp4",start_time=0)
    user_query = st.text_area(
        "What insights are you seeking from the video?",
        placeholder="Ask anything about the video content. The AI agent will analyze and gather additional context if needed.",
        help="Provide specific questions or insights you want from the video."
    )

    if st.button("🔍 Analyze Video", key="analyze_video_button"):
        if not user_query:
            st.warning("Please enter a question or insight to analyze the video.")
        else:
            try:
                with st.spinner("Processing video and gathering insights..."):
                    # Upload and process video file
                    processed_video = upload_file(video_path)
                    while processed_video.state.name == "PROCESSING":
                        time.sleep(1)
                        processed_video = get_file(processed_video.name)

                    # Prompt generation for analysis
                    analysis_prompt = (
                        f"""
                        Analyze the uploaded video for content and context.
                        Respond to the following query using video insights and supplementary web research:
                        {user_query}

                        Provide a detailed, user-friendly, and actionable response.
                        """
                    )

                    # AI agent processing
                    response = multimodal_Agent.run(analysis_prompt, videos=[processed_video])

                # Display the result
                st.subheader("Analysis Result")
                st.markdown(response.content)

            except Exception as error:
                st.error(f"An error occurred during analysis: {error}")
            finally:
                # Clean up temporary video file
                Path(video_path).unlink(missing_ok=True)
else:
    st.info("Upload a video file to begin analysis.")

# Customize text area height
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
