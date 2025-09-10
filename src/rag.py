from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
import os


docs = []

DOC_PATH = Path(__file__).parent.parent / 'politicas/'

for pdf in os.listdir(DOC_PATH):
    pdf = DOC_PATH / pdf
    try:
        loader = PyMuPDFLoader(pdf)
        docs.extend(loader.load())
        print(f'Success in file {pdf}')
    except Exception as e:
        print(f'Error in {pdf}: {e}')
