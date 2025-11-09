from flask import Flask, render_template, request, jsonify
from flask import send_from_directory
import os

# Import YoAdRAG from the voice app (shared logic)
from jarvis_app_voice import YoAdRAG

app = Flask(__name__, static_folder='static', template_folder='templates')

# Instantiate YoAd (this will load models; startup may take a moment)
yo = YoAdRAG()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def query():
    data = request.get_json(force=True)
    q = data.get('query', '').strip()
    if not q:
        return jsonify({'error': 'Empty query'}), 400

    # Always pass speak=False to avoid server-side TTS; client handles speech
    resp = yo.answer(q, speak=False)
    # Extract concise answer before returning
    concise = yo._extract_answer_text(resp)
    return jsonify({'answer': concise})


if __name__ == '__main__':
    port = int(os.getenv('WEB_UI_PORT', 7860))
    app.run(host='0.0.0.0', port=port)
