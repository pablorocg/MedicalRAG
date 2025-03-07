from setuptools import setup, find_packages

setup(
    name="rag-medico",
    version="0.1.0",
    description="Sistema RAG para consultas médicas con Ollama y ChromaDB",
    author="RAG Developer",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.13",
        "chromadb>=0.4.22",
        "tqdm>=4.66.0",
        "gradio>=4.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "lxml>=4.9.0",
        "nltk>=3.8.0",
        "rouge_score>=0.1.2",
        "tabulate>=0.9.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "rag-medico=app.main:main",
        ],
    },
)