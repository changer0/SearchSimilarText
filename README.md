## 查找相似文本

使用模型：

BAAI/bge-large-zh-v1.5

BAAI/bge-large-zh-v1.5

## 运行后端服务

```
cd backend
pip3 install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

后端服务测试：

http://localhost:8000/docs

目录结构：

```agsl
├── backend
│   ├── __pycache__
│   ├── app.py
│   ├── downloads
│   ├── launche.sh
│   ├── main.py
│   ├── requirements.txt
│   ├── scripts
│   ├── similarity_searcher.py
│   └── test_search_similar_text.py
├── backup
│   ├── FileA.pdf
│   ├── FileB.xlsx
│   ├── launche.sh
│   └── search_similar_text.py
└── frontend
    ├── index.html
    ├── script.js
    └── sytles.css

```
