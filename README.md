# 🧠 ChatBot Trait Analysis and Prediction (AI and Beyond)

This project is an AI-powered trait analysis chatbot built using **Django**, **MongoDB**, **Hugging Face Transformers**, and **PyTorch (GPU)**. It analyzes text input and predicts multiple human traits, such as mood, personality, conversational style, MBTI, interest level, and other social signals — all integrated into a clean web interface.

---

## 🚀 Features

- 🎭 **Multi-Trait Analysis**: Predicts Moods, MBTI, Big Five Personality, Conversational Style, Miscellaneous Traits, and Interest Level.
- 🤖 **HuggingFace Transformers**: Pretrained and fine-tuned models for high accuracy on NLP tasks.
- 🧠 **PyTorch GPU-Accelerated**: Built with CUDA 12.4 support for fast model inference.
- 🌐 **Django Backend**: Modular API structure using Django REST Framework.
- 🍃 **MongoDB + GridFS**: Stores and loads large model files and tokenizers directly from MongoDB using Djongo and PyMongo.
- 🐳 **Dockerized Deployment**: Easily deployable with Docker; models, backend, and services containerized for production.
- 📈 **Margin of Error Display**: Each trait prediction includes margin of error range for transparency.

---

## 🗂️ Project Structure

```
chatBotTraitPrediction/
│
├── APIs/                        # Modular APIs for each trait
│   ├── conversational_style.py
│   ├── moods.py
│   ├── big5Personality.py
│   ├── mbti.py
│   ├── miscellaneous.py
│   └── interest_level.py
│
├── models/                      # Trained models (loaded from MongoDB/GridFS)
│
├── main/                        # Django application logic
│   ├── views.py
│   ├── urls.py
│   └── utils.py
│
├── static/                      # Static assets
│
├── templates/                   # HTML frontend templates
│   ├── base.html
│   ├── home.html
│   ├── loading.html
│   └── ...
│
├── Dockerfile                   # Docker configuration
├── requirements.txt             # Python dependencies
└── manage.py                    # Django project entry point
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/chatbot-trait-prediction.git
cd chatbot-trait-prediction
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
```

### 3. Install Dependencies

💡 **First install PyTorch (GPU)**:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Then install remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## 🐳 Docker Deployment

Make sure Docker is installed and running.

### 1. Build and Run the Container

```bash
docker build -t chatbot-traits .
docker run -p 8000:8000 chatbot-traits
```

---

## 🌿 MongoDB Setup

1. **Use a globally hosted MongoDB Atlas instance or self-hosted MongoDB.**
2. **Enable GridFS** for model and tokenizer storage.
3. Update your `settings.py` and connection logic to use:

```python
DATABASES = {
    'default': {
        'ENGINE': 'djongo',
        'NAME': 'your_db_name',
        'ENFORCE_SCHEMA': False,
        'CLIENT': {
            'host': 'your-mongodb-uri',
            'uuidRepresentation': 'standard',
        }
    }
}
```

---

## 🧪 Testing the App

- Visit `http://localhost:8000`
- Enter a 20-word text or try one of the examples.
- View all trait predictions along with margin of error ranges.
- Explore each category: Mood, MBTI, Big Five, Conversational Style, etc.

---

## 🛠 Example Input

> "The meeting went extremely well, and I appreciated your honesty, energy, and constructive feedback during the session."

---

## 📦 Requirements

Full list is in `requirements.txt`, including:

- `Django==3.2`
- `djangorestframework`
- `djongo`
- `pymongo`
- `transformers`
- `torch` (GPU)
- `scikit-learn`, `numpy`, `joblib`, `tqdm`, etc.

---

## ✨ Future Improvements

- Add LLM-powered personality adaptation for chatbot responses
- Add user session storage in MongoDB for history
- Improve UI with React/Tailwind (optional future upgrade)

---

## 📄 License

This project is licensed under the **Murad Ahmad**. Feel free to use and contribute.

---

## 🙌 Acknowledgements

- HuggingFace Transformers
- PyTorch
- MongoDB & GridFS
- Django REST Framework

---

## 📬 Contact

Created with 💡 by **Murad Ahmad**  
📧 Email: aiandbeyond194@gmail.com | muradatcorvit23@gmail.com
🌐 Portfolio:
