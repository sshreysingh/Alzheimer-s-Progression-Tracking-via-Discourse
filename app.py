from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from faster_whisper import WhisperModel
import joblib
import numpy as np
import re
import math
import os
import tempfile
import warnings
from collections import Counter
from datetime import datetime
from flask import render_template

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')


# ── Fix #11: limit upload size to 50 MB ─────────────────────
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# ── Load model ──────────────────────────────────────────────
MODEL_PATH = 'alzheimers_model.pkl'
model = joblib.load(MODEL_PATH)
print(f"✅ ML model loaded: {type(model).__name__}")

# ── Whisper: use 'tiny' (already cached). Switch to 'base' only if network available. ─
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
print("✅ Whisper model loaded")

# ── Fix #12: In-memory history (last 50 entries) ─────────────
analysis_history = []

# ── Lexicons ─────────────────────────────────────────────────
FILLERS_SINGLE = {
    'um','uh','er','ah','hmm','like','well','so','right','okay',
    'ok','yeah','basically','actually','literally'
}
FILLERS_MULTI = [
    'you know', 'i mean'
]
FILLERS = FILLERS_SINGLE | set(FILLERS_MULTI)
NEGATIONS = {
    'not','no','never','neither','nor','none','nobody','nothing',
    'nowhere','hardly','barely','scarcely'
}
PRONOUNS = {
    'i','me','my','mine','myself','you','your','yours','he','him',
    'his','she','her','hers','it','its','we','us','our','ours',
    'they','them','their','theirs','this','that','these','those'
}
CONJUNCTIONS = {
    'and','but','or','nor','for','yet','so','although','because',
    'since','while','after','before','if','though','unless','until',
    'when','where','whereas','whether'
}
FUNCTION_WORDS = {
    'the','a','an','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','will','would','could',
    'should','may','might','must','shall','can','to','of','in',
    'on','at','by','for','with','as','it','and','but','or',
    'i','you','he','she','we','they','this','that','not','there','here'
}
VERBS = {
    'is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might',
    'must','shall','can','get','go','make','seem','look','try','run',
    'come','see','know','think','want','take','give','put','fall',
    'blow','wipe','stand','sit','going','getting','trying','running',
    'wiping','standing','sitting','coming','seeing','knowing',
    'thinking','wanting','taking','giving','putting','falling','blowing'
}
IU_KEYWORDS = [
    'cookie','cookies','jar','stool','tipping','tippling','tip','boy',
    'girl','woman','mother','mom','lady','sink','water','overflow',
    'overflowing','dish','dishes','plate','wipe','wiping','drying',
    'kitchen','window','curtain','curtains','outside','garden','yard',
    'fall','falling','stealing','reaching','cabinet','counter','floor',
    'apron','cup','stepping'
]

# ── Feature name list (order matches training) ───────────────
FEATURE_NAMES = [
    'n_words', 'n_sentences', 'ttr', 'honore', 'brunet',
    'neg_rate', 'revision_count', 'verb_rate', 'mean_sent_len',
    'sent_len_var', 'iu_count', 'pronoun_rate', 'conj_rate',
    'filler_rate', 'mean_word_len', 'repetition_rate',
    'iu_density', 'unique_words'
]


def extract_linguistic_features(text: str) -> np.ndarray:
    """Extract the 18 linguistic features the model was trained on."""
    text = str(text).strip()

    sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    n_sentences = max(len(sents), 1)

    words = re.findall(r"\b\w+(?:'\w+)?\b", text)
    n_words = max(len(words), 1)
    wl = [w.lower() for w in words]

    freq = Counter(wl)
    unique_words = len(freq)
    ttr = unique_words / n_words
    v1 = sum(1 for c in freq.values() if c == 1)
    v = unique_words
    denom = (1 - v1 / v) if v1 < v else 1e-9
    honore = 100 * math.log(n_words + 1) / denom
    brunet = n_words ** (v ** -0.165) if v > 0 else 0.0

    sl = [len(re.findall(r"\b\w+(?:'\w+)?\b", s)) for s in sents]
    mean_sent_len = float(np.mean(sl)) if sl else 0.0
    sent_len_var = float(np.var(sl)) if len(sl) > 1 else 0.0

    mean_word_len = float(np.mean([len(w) for w in words])) if words else 0.0

    verb_rate       = sum(1 for w in wl if w in VERBS)          / n_words
    pronoun_rate    = sum(1 for w in wl if w in PRONOUNS)       / n_words
    conj_rate       = sum(1 for w in wl if w in CONJUNCTIONS)   / n_words
    neg_rate        = sum(1 for w in wl if w in NEGATIONS)      / n_words
    filler_count    = sum(1 for w in wl if w in FILLERS_SINGLE)
    filler_count   += sum(text.lower().count(p) for p in FILLERS_MULTI)
    filler_rate     = filler_count / n_words

    reps = sum(1 for i in range(1, len(wl)) if wl[i] == wl[i - 1])
    repetition_rate = reps / n_words
    revision_count  = sum(
        1 for p in ['or rather', 'i mean', 'that is', 'i meant']
        if p in text.lower()
    ) / n_words

    tl = text.lower()
    iu_count = sum(1 for kw in IU_KEYWORDS if kw in tl)
    iu_density = iu_count / (n_words * n_sentences)

    features = np.array([
        n_words, n_sentences, ttr, honore, brunet,
        neg_rate, revision_count, verb_rate, mean_sent_len,
        sent_len_var, iu_count, pronoun_rate, conj_rate,
        filler_rate, mean_word_len, repetition_rate,
        iu_density, unique_words,
    ], dtype=float)

    return features




@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    suffix = os.path.splitext(file.filename)[-1] or '.wav'

    patient_id   = request.form.get('patient_id', '')
    patient_name = request.form.get('patient_name', '')
    patient_age  = request.form.get('patient_age', '')
    notes        = request.form.get('notes', '')

    # ── Fix #10: init tmp_path before try so finally is safe ─
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        # Step 1: Transcribe
        segments, info = whisper_model.transcribe(tmp_path, beam_size=5)
        transcript = ' '.join(seg.text.strip() for seg in segments).strip()

        if not transcript:
            return jsonify({'error': 'Could not transcribe audio. Please speak clearly.'}), 400

        # ── Fix #4: reject audio shorter than 3 seconds ──────
        if info.duration < 3.0:
            return jsonify({
                'error': f'Audio too short ({info.duration:.1f}s). Please record at least 3 seconds.'
            }), 400

        # Step 2: Extract features
        features = extract_linguistic_features(transcript)
        X = features.reshape(1, -1)

        # Step 3: Predict
        prediction    = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        # ── Fix #5: return named feature breakdown ────────────
        features_raw = {
            name: round(float(val), 4)
            for name, val in zip(FEATURE_NAMES, features)
        }

        result = {
            'prediction':    int(prediction),
            'label':         "Alzheimer's Detected" if prediction == 1 else "No Alzheimer's Detected",
            'confidence':    round(float(probabilities[prediction]) * 100, 1),
            'prob_positive': round(float(probabilities[1]) * 100, 1),
            'prob_negative': round(float(probabilities[0]) * 100, 1),
            'transcript':    transcript,
            'features_raw':  features_raw,
            'audio_info': {
                'duration_seconds':   round(info.duration, 2),
                'language':           info.language,
                'features_extracted': len(features)
            },
            'patient': {
                'id':    patient_id,
                'name':  patient_name,
                'age':   patient_age,
                'notes': notes
            },
            'timestamp': datetime.now().isoformat()
        }

        # ── Fix #12: store in history ─────────────────────────
        analysis_history.append(result)
        if len(analysis_history) > 50:
            analysis_history.pop(0)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # ── Fix #10: safe cleanup even if save never happened ─
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route('/history', methods=['GET'])
def history():
    """Return the last 20 analyses in reverse-chronological order."""
    return jsonify(list(reversed(analysis_history[-20:])))


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': type(model).__name__,
        'history_count': len(analysis_history)
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
