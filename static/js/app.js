// Client-side JS for YoAd web UI
const queryEl = document.getElementById('query');
const btnSend = document.getElementById('btn-send');
const btnSpeak = document.getElementById('btn-speak');
const answerEl = document.getElementById('answer');

let recognizing = false;
let recognition = null;

if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SR();
  recognition.lang = 'en-US';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  recognition.onresult = (event) => {
    const text = event.results[0][0].transcript;
    queryEl.value = text;
  };
  recognition.onerror = (event) => {
    console.warn('Speech recognition error', event);
  };
  recognition.onend = () => {
    recognizing = false;
    btnSpeak.textContent = '🎤 Start Voice';
  };
}

btnSpeak.addEventListener('click', () => {
  if (!recognition) {
    alert('Speech recognition not supported in this browser.');
    return;
  }

  if (!recognizing) {
    recognition.start();
    recognizing = true;
    btnSpeak.textContent = '⏹ Stop';
  } else {
    recognition.stop();
    recognizing = false;
    btnSpeak.textContent = '🎤 Start Voice';
  }
});

async function sendQuery(q) {
  answerEl.textContent = 'Thinking...';
  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q }),
    });
    const data = await res.json();
    if (data.error) {
      answerEl.textContent = 'Error: ' + data.error;
      return;
    }
    answerEl.textContent = data.answer;

    // Speak the answer using browser TTS
    if ('speechSynthesis' in window) {
      const u = new SpeechSynthesisUtterance(data.answer);
      u.lang = 'en-US';
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(u);
    }
  } catch (e) {
    answerEl.textContent = 'Request failed: ' + e;
  }
}

btnSend.addEventListener('click', () => {
  const q = queryEl.value.trim();
  if (!q) return;
  sendQuery(q);
});

// Allow Enter to send (Shift+Enter for newline)
queryEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    btnSend.click();
  }
});
