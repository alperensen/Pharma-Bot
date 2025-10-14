🚀 Kurulum
1. Gerekli Paketleri Yükleyin
# Virtual environment oluşturun (opsiyonel ama önerilir)
python3 -m venv genai-env
source genai-env/bin/activate  # macOS/Linux
# genai-env\Scripts\activate  # Windows

# Paketleri yükleyin
pip install -r requirements.txt
2. API Anahtarlarını Ayarlayın
Proje kök dizininde .env dosyası oluşturun:

GOOGLE_API_KEY=your_google_api_key_here
HF_TOKEN=your_huggingface_token_here
Google API Key: Google AI Studio üzerinden alabilirsiniz
Hugging Face Token: Hugging Face Settings üzerinden alabilirsiniz
Veri setine erişim için: turkish-academic-theses-dataset sayfasından erişim izni isteyin


pip install langchain sentence-transformers faiss-cpu "transformers>=4.32.0"

pip install -U langchain-community