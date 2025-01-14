import streamlit as st
from scipy.io.wavfile import write
import numpy as np
from groq import Groq
import os
import tempfile
import markdown
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io
import wave
from st_audiorec import st_audiorec

# Ensure the necessary packages are installed
os.system('pip install streamlit-audiorec')

def get_audio_devices():
    """Get list of available audio input devices"""
    try:
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append((i, device['name']))
        return input_devices
    except Exception as e:
        return []

def transcribe_text(file_path, client):
    """Transcribe audio file to text using Groq"""
    try:
        with open(file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(file_path, file.read()),
                model="whisper-large-v3-turbo",
                response_format="json",
                language="en",
                temperature=0.0
            )
            return transcription.text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def convert_chunk_to_notes(client, chunk):
    """Convert text chunk to structured notes"""
    prompt = f"""
    Convert the following text into well-structured lecture notes. Include:
    - Main topics and subtopics
    - Key concepts and definitions
    - Any formulas or equations mentioned
    - Brief summaries of main points
    - Important questions and answers in separate lines based on the audio provided

    Text to convert:
    {chunk}

    Format the notes in a clear, hierarchical structure using markdown.
    Do not mention unnecessary details.
    """

    try:
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at converting text into clear, concise lecture notes."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating notes: {str(e)}"

def convert_md_to_pdf(md_text):
    """Convert markdown text to PDF"""
    # Convert markdown to HTML
    html = markdown.markdown(md_text)
    
    # Create PDF in memory
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    story = []
    
    for part in html.split('\n'):
        if part.strip():
            if part.startswith('<h1>'):
                style = styles['Heading1']
            elif part.startswith('<h2>'):
                style = styles['Heading2']
            elif part.startswith('<h3>'):
                style = styles['Heading3']
            else:
                style = styles['Normal']
            
            text = part.replace('<p>', '').replace('</p>', '')
            text = text.replace('<h1>', '').replace('</h1>', '')
            text = text.replace('<h2>', '').replace('</h2>', '')
            text = text.replace('<h3>', '').replace('</h3>', '')
            
            story.append(Paragraph(text, style))
            story.append(Spacer(1, 12))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def main():
    st.title("NOTEIFY :pencil:")
    Markdown = '''
    ## Say it loud, See it written! :wink:
    '''
    st.write(Markdown)
    st.write("Crafted with :bulb: by Shreya Khandelwal, Ditya Goel, Mridunja Raman, Riddhi Motwani and Preksha.")

    # Sidebar for API key
    api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
    instructions=st.sidebar.write("Here’s a quick setup guide for the Groq API:")
    instructions1=st.sidebar.write("1. Open https://groq.com")
    instructions2=st.sidebar.write("2. On the top-right corner, click on 'DEV CONSOLE'")
    instructions3=st.sidebar.write("3. On the sidebar, click on API keys and Create a New API Key.")
    instructions1=st.sidebar.write("4. The secret key is your Groq API key. Copy it and paste it in the input box.")
    
    
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar to continue.")
        return
    
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
        return
    
    # Main interface
    st.header("1. Record or Upload Audio")
    
    option = st.radio("Choose input method:", ["Record Audio", "Upload Audio File"])
    
    if option == "Record Audio":
        # Call the audio recorder
        audio_data = st_audiorec()
        
        # If audio_data is not None, save it as a temporary .wav file
        if audio_data is not None:
            # Create a temporary file to save the audio data
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(audio_data)
                audio_file_path = temp_file.name  # Get the file path of the saved audio
                
                # Display audio data
                st.audio(audio_file_path, format="audio/wav")
                
                # Store the file path in session_state for later processing
                st.session_state['audio_file'] = audio_file_path

        # Transcription and Notes Generation
        if 'audio_file' in st.session_state and st.button("Generate Notes"):
            try:
                with st.spinner("Transcribing audio..."):
                    text = transcribe_text(st.session_state['audio_file'], client)
                    st.text_area("Transcribed Text", text, height=200)
                
                with st.spinner("Generating notes..."):
                    chunks = [text]  # Simplified for this example
                    notes_sections = []
                    
                    for chunk in chunks:
                        notes = convert_chunk_to_notes(client, chunk)
                        if notes:
                            notes_sections.append(notes)
                    
                    final_notes = "\n\n".join(notes_sections)
                    st.markdown("### Generated Notes")
                    st.markdown(final_notes)
                    
                    # Create download buttons
                    st.download_button(
                        label="Download Notes (Markdown)",
                        data=final_notes,
                        file_name="lecture_notes.md",
                        mime="text/markdown"
                    )
                    
                    # Convert to PDF and offer download
                    pdf_buffer = convert_md_to_pdf(final_notes)
                    st.download_button(
                        label="Download Notes (PDF)",
                        data=pdf_buffer,
                        file_name="lecture_notes.pdf",
                        mime="application/pdf"
                    )
                    
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
            
            finally:
                # Cleanup temporary files
                if 'audio_file' in st.session_state:
                    try:
                        os.unlink(st.session_state['audio_file'])
                    except:
                        pass

    else:  # Upload Audio File
        uploaded_file = st.file_uploader("Choose an audio file", type=['wav'])
        if uploaded_file:
            # Save uploaded file to temporary file
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_audio.write(uploaded_file.read())
            st.session_state['audio_file'] = temp_audio.name
            st.audio(uploaded_file)
    
        # Transcription and Notes Generation
        if 'audio_file' in st.session_state and st.button("Generate Notes"):
            try:
                with st.spinner("Transcribing audio..."):
                    text = transcribe_text(st.session_state['audio_file'], client)
                    st.text_area("Transcribed Text", text, height=200)
                
                with st.spinner("Generating notes..."):
                    chunks = [text]  # Simplified for this example
                    notes_sections = []
                    
                    for chunk in chunks:
                        notes = convert_chunk_to_notes(client, chunk)
                        if notes:
                            notes_sections.append(notes)
                    
                    final_notes = "\n\n".join(notes_sections)
                    st.markdown("### Generated Notes")
                    st.markdown(final_notes)
                    
                    # Create download buttons
                    st.download_button(
                        label="Download Notes (Markdown)",
                        data=final_notes,
                        file_name="lecture_notes.md",
                        mime="text/markdown"
                    )
                    
                    # Convert to PDF and offer download
                    pdf_buffer = convert_md_to_pdf(final_notes)
                    st.download_button(
                        label="Download Notes (PDF)",
                        data=pdf_buffer,
                        file_name="lecture_notes.pdf",
                        mime="application/pdf"
                    )
                    
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
            
            finally:
                # Cleanup temporary files
                if 'audio_file' in st.session_state:
                    try:
                        os.unlink(st.session_state['audio_file'])
                    except:
                        pass

if __name__ == "__main__":
    main()
