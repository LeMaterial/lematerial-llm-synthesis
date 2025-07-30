from datasets import load_dataset, Dataset, DatasetDict, Features
import argparse
import torch
from pathlib import Path
from paper_schema import schema
import requests
from marker.converters.pdf import PdfConverter
from marker.output import text_from_rendered
from marker.models import create_model_dict
from io import BytesIO
import re
from multiprocess import set_start_method
from scidownl import scihub_download

converter = PdfConverter(artifact_dict=create_model_dict())

journal_abbrs = {
    '2076-3417': 'app',   # Applied Sciences
    '1996-1944': 'ma',    # Materials
    '2073-4352': 'crystals',      # Crystals
    '2311-5629': 'micromachines', # Micromachines
    '2310-2861': 'gels',          # Gels
    '2227-9717': 'processes',     # Processes
    '1996-1073': 'energies',      # Energies
    '2079-4991': 'nanomaterials', # Nanomaterials
    '2304-6740': 'inorganics',    # Inorganics
    '2673-4583': 'chemengineering', # ChemEngineering
    '2079-6412': 'coatings',      # Coatings
    '1422-0067': 'ijms',          # International Journal of Molecular Sciences
    '2073-4360': 'polymers',      # Polymers
    '2073-4344': 'catalysts',     # Catalysts
    '1999-4923': 'pharmaceutics', # Pharmaceutics
    '2075-4701': 'metals',        # Metals
}

class ImageTextExtractor(object):
    def __init__(self, args):
        self.dataset = load_dataset(args.dataset, name=args.config, split=args.split)
        # should remove line after this unless youre sure
        self.dataset = self.dataset.cast(Features.from_arrow_schema(schema))
        self.converter = PdfConverter(artifact_dict=create_model_dict())
        self.args = args

        self.output_dir = Path(args.pdf_dir)
        self.output_dir.mkdir(exist_ok=True)

    def pil_to_bytes(self, pil_img):
        buf = BytesIO()
        pil_img.save(buf, format="JPEG")    # choose the format we need
        raw_bytes = buf.getvalue()  
        buf.close()
        return raw_bytes

    def process_row(self, row):
        if self.args.skip_if_processed:
            if row['images'] is not None:
                print(row['images'][0]['path'])
                return row

        url = row['pdf_url']

        # where to write pdf
        filename = url.split("/")[-1]
        if not filename.endswith(".pdf"):
            filename += ".pdf"

        file_path = self.output_dir / filename

        headers={"User-Agent": "Mozilla/5.0"}
        try:
            # try getting it from pdf link
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(response.content)
            rendered = converter(str(file_path))
        except:
            # if blocked, try getting it from scihub
            try:
                if 'mdpi' in url:
                    pattern = r'mdpi\.com/([\d\-]+)/(\d+)/(\d+)/(\d+)/'
                    match = re.search(pattern, url)
                    if match:
                        journal_id, volume, issue, article = match.groups()

                        # map journal_id to abbreviation
                        journal_abbr = journal_abbrs.get(journal_id, 'unknown')
                        # if unknown, it will fail for row, but not crash
                        doi = f'10.3390/{journal_abbr}{volume}{issue}{article}'
                elif 'rsc' in url:
                    match = re.search(r'/([^/]+)$', url)
                    if match:
                        manuscript_id = match.group(1)
                    doi = f'10.1039/{manuscript_id}'
               
                scihub_download(doi, paper_type='doi', out=str(file_path))
                rendered = converter(str(file_path))
            except:
                print("failed on: ", url)
                return row

        print(f"Downloaded: {file_path}")
        text, _, images = text_from_rendered(rendered)
        

        row['text_paper'] = text
        row['images'] = [
                        {"path": path, "bytes": self.pil_to_bytes(pil_img)}
                        for path, pil_img in images.items()
                    ]

        return row
    
    def extract_all(self):
        if args.multiprocess:
            set_start_method("spawn")
            print("num_proc: ", torch.cuda.device_count())
            enhanced_dataset = self.dataset.map(self.process_row, num_proc=torch.cuda.device_count())
        else:
            enhanced_dataset = enhanced_dataset.cast(Features.from_arrow_schema(schema))


        if self.args.write_to_disk:
            enhanced_dataset.save_to_disk(args.disk_location)

        if self.args.write_to_hub:
            enhanced_dataset.push_to_hub(self.args.dataset, config_name=self.args.config, split=self.args.split, create_pr=True)
           

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write_to_hub", action="store_true", default=True, help="do we write to the remote dataset?")
    parser.add_argument("--write_to_disk", action="store_true", default=True)
    parser.add_argument("--disk_location", type=str, default='/fsx/georgia_channing/temp')
    parser.add_argument("--skip_if_processed", default=False, help='If the row already had images, skip processing it again.')
    parser.add_argument("--dataset", type=str, default='LeMaterial/LeMat-Synth-Papers')
    parser.add_argument("--pdf_dir", type=str, default='pdfs', help='where to write PDFs we download')
    parser.add_argument("--config", type=str, default='default')
    parser.add_argument("--split", type=str, required=True, default='sample_for_evaluation')
    parser.add_argument("--multiprocess", type=bool, default=True)
    args = parser.parse_args()

    ImageTextExtractor(args=args).extract_all()