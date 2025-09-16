# Demo Yolo + Recommendation System

### Tạo môi trường ảo
```bash
python -m venv venv
source venv/bin/activate  # Trên macOS/Linux
venv\Scripts\activate  # Trên Windows
```

### Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Chạy Backend Recommendation System
```bash
uvicorn app:app --reload
```

### Chạy UI demo
```bash
streamlit run app.py
```
