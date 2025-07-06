import gradio as gr
import whisper
import openai
import tempfile

openai.api_key = "your-openai-key-here"  # Replace with your key or use os.getenv()

model = whisper.load_model("base")

def transcribe_and_summarize(audio):
    if audio is None:
        return "No audio provided.", ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio.read())
        tmp_path = tmp.name

    result = model.transcribe(tmp_path)
    transcription = result["text"]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You're a helpful assistant."},
            {"role": "user", "content": f"Summarize this meeting: {transcription}"}
        ],
    )
    summary = response.choices[0].message["content"]
    return transcription, summary

interface = gr.Interface(
    fn=transcribe_and_summarize,
    inputs=gr.Audio(source="upload", type="file", label="Upload audio (WAV/MP3)"),
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="Summary")
    ],
    title="🎙️ Meeting Transcriber + Summarizer",
    description="Upload an audio file to transcribe and summarize it using Whisper + GPT."
)

if __name__ == "__main__":
    interface.launch()
