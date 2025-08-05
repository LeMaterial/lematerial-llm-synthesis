import requests
from bs4 import BeautifulSoup
import html2text
import tarfile
import re
from pathlib import Path
import os
import pypandoc
import shutil

class ArxivScraper():
    def __init__(self, temp_dir=None):
        self.base_url = 'http://export.arxiv.org/api/query'
        self.html_url = 'https://arxiv.org/html/'
        self.src_url = 'https://arxiv.org/src/' # this returns latex
        self.temp_dir = temp_dir if temp_dir is not None else '.'

    def parse_latex(self, response):
        arxiv_id = response.url.split('/')[-1]
        extract_dir = os.path.join(self.temp_dir, arxiv_id+'_extracted')
        tar_file = os.path.join(self.temp_dir, arxiv_id+'.tar.gz')
        with open(tar_file, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)

        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(extract_dir)

        main_tex = None
        image_files = {}
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf', '.eps')):
                    # base = os.path.splitext(file)[0]
                    image_files[file] = os.path.abspath(os.path.join(root, file))
                if file.endswith(".tex"):
                    main_tex = os.path.join(root, file)

        with open(main_tex, 'r', encoding='utf-8') as f:
            print(main_tex)
            content = f.read()

        try:
            markdown_text = pypandoc.convert_text(content, 'gfm', format='latex+raw_tex', extra_args=['--mathjax'])
        except:
            os.remove(tar_file)
            shutil.rmtree(extract_dir)
            return None, None

        image_data = {}
        for name, path in image_files.items():
            with open(path, "rb") as img_file:
                image_data[name] = img_file.read()

        # clean up
        os.remove(tar_file)
        shutil.rmtree(extract_dir)
        
        return markdown_text, image_data
    
    def parse_html(self, response):
        return None, None
    
    def parse_pdf(self, response):
        return None, None

    def parse_from_id(self, id):
        # try html
        response = requests.get(self.html_url+id)
        soup = BeautifulSoup(response.text, "html.parser")
        text = None
        images = None
        if "No HTML for" in soup.text:
            response = requests.get(self.src_url+id)
            content_type = response.headers.get("Content-Type", "")
            if 'gzip' in content_type:
                try:
                    text, images = self.parse_latex(response)
                except:
                    print("failed for id: ", id)
                    pass
                method = 'latex'
            elif 'pdf' in content_type:
                text, images = self.parse_pdf(response)
                method = 'pdf'
            else:
                raise ValueError("The response from arxiv is not PDF, HTML, or Latex.")
        else:
            method = 'html'
            text, images = self.parse_html(response)
                
        return text, images, method

# ArxivAPI().parse_from_id(id='2007.02129')

