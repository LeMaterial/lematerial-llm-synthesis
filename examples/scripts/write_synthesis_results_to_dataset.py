from datasets import load_dataset, Dataset, DatasetDict, Features
import argparse
import pandas as pd
from pathlib import Path
import json
from synthesis_schema import schema


class SynthesisWriter(object):
    def __init__(self, args):
        self.paper_df = load_dataset(args.paper_dataset, name=args.config, split=args.split, columns=['id', 'title', 'abstract', 'doi', 'pdf_url', 'images', 'published_date']).to_pandas()
        self.args = args

    def filter_images(self, images):
        return images, None

    def merge_information(self, row):
        new_row = {}
        # make sure the paper id is in the paper dataframe
        if row['id'] not in self.paper_df['id'].values:
            return
        new_row['synthesized_material'] = row['synthesis']['target_compound']
        new_row['material_category'] = row['synthesis']['target_compound_type']
        new_row['synthesis_method'] = row['synthesis']['synthesis_method']
        new_row['images'], new_row['plot_data'] = self.filter_images(self.paper_df[self.paper_df['id'] == row['id']].iloc[0]['images'])
        new_row['structured_synthesis'] = row['synthesis']
        new_row['evaluation'] = row['evaluation']
        for col in ['synthesis_extraction_performance_llm', 'figure_extraction_performance_llm', 'synthesis_extraction_performance_human', 'figure_extraction_performance_human']:
            new_row[col] = None
        new_row['paper_title'] = self.paper_df[self.paper_df['id'] == row['id']].iloc[0]['title']
        new_row['paper_published_date'] = self.paper_df[self.paper_df['id'] == row['id']].iloc[0]['published_date']
        new_row['paper_abstract'] = self.paper_df[self.paper_df['id'] == row['id']].iloc[0]['abstract']
        new_row['paper_doi'] = self.paper_df[self.paper_df['id'] == row['id']].iloc[0]['doi']
        new_row['paper_url'] = self.paper_df[self.paper_df['id'] == row['id']].iloc[0]['pdf_url']
        return new_row

    def extract_synthesis_recipes(self):
        results_dir = Path(self.args.results_dir)
        all_records = []

        
        for json_path in results_dir.rglob("result.json"):
            with open(json_path) as f:
                records = json.load(f)
                if isinstance(records, list):
                    top_level = json_path.relative_to(results_dir)
                    for record in records:
                        record["id"] = str(top_level).strip('/result.json')
                        all_records.append(record)
                else:
                    print(f"Skipping non-list JSON in {json_path}")

        if not all_records:
            raise ValueError("It appears the results directory you have passed is empty or contains no files called result.json in subdirectories.")

        results_df = pd.DataFrame(all_records)
        merged_records = results_df.apply(self.merge_information, axis=1).dropna().to_list()
        final_df = pd.DataFrame(merged_records).drop_duplicates(subset=['paper_url', 'synthesized_material'])

        if args.write_to_hub:
            hf_dataset = Dataset.from_pandas(final_df, preserve_index=False, features=Features.from_arrow_schema(schema))
            hf_dataset.push_to_hub(self.args.synthesis_dataset, config_name=self.args.config, split=self.args.split, create_pr=True)

        return 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--write_to_hub", action="store_true", default=True, help="do we write to the remote dataset?")
    parser.add_argument("--results_dir", type=str, default="../../results")
    parser.add_argument("--paper_dataset", type=str, default='LeMaterial/LeMat-Synth-Papers')
    parser.add_argument("--synthesis_dataset", type=str, default='LeMaterial/LeMat-Synth')
    parser.add_argument("--config", type=str, required=True, default='default')
    parser.add_argument("--split", type=str, required=True, default='sample_for_evaluation')
    args = parser.parse_args()

    SynthesisWriter(args=args).extract_synthesis_recipes()