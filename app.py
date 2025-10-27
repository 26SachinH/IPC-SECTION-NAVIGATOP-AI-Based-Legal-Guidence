from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer, util
from gtts import gTTS
import torch
import os
import pandas as pd
from googletrans import Translator
import uuid

app = Flask(__name__)

# Load model and IPC dataset
model = SentenceTransformer("ipc-multilang-model-2")
df = pd.read_csv(r"C:\Users\SACHIN HEBBALAKAR\OneDrive\Desktop\major project\IPC SECTION\ipc_sections_converted.csv")
df = df.dropna(subset=['section_desc']).reset_index(drop=True)
df['combined'] = df['Section'] + " | " + df['section_title'] + " | " + df['section_desc']
ipc_embeddings = model.encode(df['combined'], convert_to_tensor=True)

translator = Translator()

# Translation functions
def translate_to_english(text):
    try:
        translated = translator.translate(text, dest='en')
        print(f"[Translation] Detected: {translated.src}, Translated: {translated.text}")
        return translated.text, translated.src
    except Exception as e:
        print(f"[Error] translate_to_english: {e}")
        return text, 'en'

def translate_to_language(text, lang_code='hi'):
    try:
        if len(text) > 500:
            parts = text.split('. ')
            translated = [translator.translate(p, dest=lang_code).text for p in parts if p.strip()]
            return '. '.join(translated)
        return translator.translate(text, dest=lang_code).text
    except Exception as e:
        print(f"[Error] translate_to_language: {e}")
        return text

# Audio generation
def generate_audio(text, lang='en'):
    if not os.path.exists('static'):
        os.makedirs('static')
    filename = f"output_{uuid.uuid4().hex[:8]}.mp3"
    filepath = os.path.join('static', filename)
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save(filepath)
        return filepath
    except Exception as e:
        print(f"[Error] generate_audio: {e}")
        return None

# Recommendation
def get_ipc_recommendations_multilang(query, detected_lang, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, ipc_embeddings)[0]
    top_results = torch.topk(scores, k=top_k)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        section = df.iloc[idx.item()]
        result = {
            'Section': section['Section'],
            'Title': section['section_title'],
            'Description': section['section_desc'],
            'Score': round(score.item(), 4)
        }

        if detected_lang != 'en':
            result['Title'] = translate_to_language(result['Title'], lang_code=detected_lang)
            result['Description'] = translate_to_language(result['Description'], lang_code=detected_lang)

        results.append(result)

    return results

# Home page route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_query = request.form['query']
        selected_lang = request.form.get('lang', 'auto')

        # Detect or use selected language
        if selected_lang != 'auto':
            detected_lang = selected_lang
            translated_query = translate_to_english(user_query)[0] if selected_lang != 'en' else user_query
        else:
            translated_query, detected_lang = translate_to_english(user_query)

        results = get_ipc_recommendations_multilang(translated_query, detected_lang)

        # Remove old audio files
        for f in os.listdir("static"):
            if f.startswith("output_") and f.endswith(".mp3"):
                try:
                    os.remove(os.path.join("static", f))
                except:
                    pass

        audio_path = None
        if results:
            text_to_speak = (
                f"Section {results[0]['Section']}. "
                f"Title: {results[0]['Title']}. "
                f"Description: {results[0]['Description']}"
            )
            audio_path = generate_audio(text_to_speak, lang=detected_lang if detected_lang in ['en', 'hi', 'kn'] else 'en')

        return render_template("index.html", results=results, query=translated_query,
                               detected_lang=detected_lang, audio_path=audio_path)

    return render_template("index.html", results=[], query=None, detected_lang=None, audio_path=None)

# New screenshots page route
@app.route('/screenshots')
def screenshots():
    return render_template('screenshots.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
