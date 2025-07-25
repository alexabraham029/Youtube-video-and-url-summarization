import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import os
import re

## Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

## Get API Keys
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    if "youtube.com/watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

def get_youtube_transcript(video_id):
    """Get YouTube transcript using multiple methods"""
    try:
        # Method 1: Try youtube-transcript-api
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
            text = " ".join([item['text'] for item in transcript_list])
            return text, "transcript-api"
        except ImportError:
            st.warning("youtube-transcript-api not installed. Install with: pip install youtube-transcript-api")
        except Exception as e:
            st.warning(f"Transcript API failed: {str(e)}")
    
        # Method 2: Try yt-dlp
        try:
            import yt_dlp
            ydl_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en'],
                'skip_download': True,
                'quiet': True,
                'no_warnings': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://youtube.com/watch?v={video_id}", download=False)
                
                # Get description as fallback
                description = info.get('description', '')
                if description and len(description) > 100:
                    return description[:2000], "description"  # Limit length
                    
        except ImportError:
            st.warning("yt-dlp not installed. Install with: pip install yt-dlp")
        except Exception as e:
            st.warning(f"yt-dlp failed: {str(e)}")
    
        return None, None
        
    except Exception as e:
        return None, str(e)

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL. It can be a YouTube video URL or website URL")
    else:
        try:
            with st.spinner("Loading content..."):
                # Initialize ChatGroq
                llm = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
                
                docs = None
                
                ## Loading YouTube or website data
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    st.info("🔄 Loading YouTube video...")
                    
                    # Extract video ID
                    video_id = extract_video_id(generic_url)
                    if not video_id:
                        st.error("❌ Could not extract video ID from URL")
                    else:
                        # Method 1: Try original YoutubeLoader (most reliable when it works)
                        try:
                            loader = YoutubeLoader.from_youtube_url(
                                generic_url,
                                add_video_info=True,
                                language=['en', 'en-US']
                            )
                            docs = loader.load()
                            st.success("✅ Loaded using YoutubeLoader")
                            
                        except Exception as e:
                            st.warning(f"YoutubeLoader failed: {str(e)}")
                            
                            # Method 2: Try transcript extraction
                            transcript_text, method = get_youtube_transcript(video_id)
                            
                            if transcript_text:
                                # Create document manually
                                class SimpleDoc:
                                    def __init__(self, content, metadata=None):
                                        self.page_content = content
                                        self.metadata = metadata or {}
                                
                                docs = [SimpleDoc(
                                    transcript_text,
                                    {"source": generic_url, "method": method}
                                )]
                                st.success(f"✅ Loaded using {method}")
                            else:
                                st.error("❌ All YouTube loading methods failed")
                                st.info("""
                                **Troubleshooting YouTube Issues:**
                                1. Make sure the video is public
                                2. Check if the video has captions/transcripts
                                3. Install required packages:
                                   - `pip install youtube-transcript-api`
                                   - `pip install yt-dlp`
                                4. Try a different YouTube URL
                                """)
                
                else:
                    # Website loading
                    st.info("🔄 Loading website...")
                    try:
                        loader = UnstructuredURLLoader(
                            urls=[generic_url],
                            ssl_verify=False,
                            headers={
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                            }
                        )
                        docs = loader.load()
                        st.success("✅ Website loaded successfully")
                    except Exception as e:
                        st.error(f"❌ Failed to load website: {str(e)}")

                # Generate summary if content was loaded
                if docs and len(docs) > 0:
                    with st.spinner("🤖 Generating summary..."):
                        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                        output_summary = chain.run(docs)
                        
                        st.subheader("📋 Summary")
                        st.success(output_summary)
                        
                        # Show content details
                        with st.expander("📊 Content Details"):
                            st.write(f"**Content length:** {len(docs[0].page_content)} characters")
                            if hasattr(docs[0], 'metadata') and docs[0].metadata:
                                st.json(docs[0].metadata)
                else:
                    st.error("❌ Could not load any content from the URL")
                    
        except Exception as e:
            error_msg = str(e).lower()
            if "groq_api_key" in error_msg or "api key" in error_msg:
                st.error("❌ Invalid Groq API key. Please check your key and try again.")
            elif "quota" in error_msg:
                st.error("❌ API quota exceeded. Please try again later.")
            else:
                st.error(f"❌ Unexpected error: {str(e)}")

# Helpful information
if not groq_api_key:
    st.info("👈 Enter your Groq API key in the sidebar to get started!")
    st.markdown("""
    **Get your free Groq API key:**
    1. Go to [console.groq.com](https://console.groq.com)
    2. Sign up for free
    3. Create an API key
    4. Paste it in the sidebar
    """)

# Installation instructions
with st.expander("📦 Installation Requirements"):
    st.code("""
    pip install youtube-transcript-api
    pip install yt-dlp
    """)

# Example URLs
with st.expander("📝 Example URLs to try"):
    st.markdown("""
    **YouTube Videos (with captions):**
    - TED Talks: `https://www.youtube.com/watch?v=UyyjU8fzEYU`
    - Educational content: `https://www.youtube.com/watch?v=aircAruvnKk`
    
    **Websites:**
    - `https://en.wikipedia.org/wiki/Artificial_intelligence`
    - `https://blog.openai.com/chatgpt`
    - `https://www.bbc.com/news`
    """)