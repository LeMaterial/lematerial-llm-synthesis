import asyncio
import os
import warnings
from urllib.parse import urlparse

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from tqdm import tqdm

warnings.filterwarnings("ignore")
load_dotenv()
USER = os.environ.get("USER", "mlederbau")
DATA_DIR = f"/Users/{USER}/lematerial-llm-synthesis/data/"
PDFS_DIR = os.path.join(DATA_DIR, "pdfs_omg24")
HUGGINGFACE_DATASET = "magdaroni/chemrxiv-dev"
SPLIT = "filtered_omg24"
BATCH_SIZE = 1


def ensure_directory(path: str):
    os.makedirs(path, exist_ok=True)


async def download_file_with_playwright(
    browser, url: str, dirpath: str, filename: str
) -> str:
    """Private helper method to download a file from a URL using Playwright."""
    out_path = os.path.join(dirpath, filename)
    if os.path.exists(out_path):
        print(f"File already exists: {out_path}")
        return out_path

    context = None
    page = None
    try:
        user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
        
        context = await browser.new_context(
            user_agent=user_agent,
            viewport={"width": 1920, "height": 1080},
            accept_downloads=True
        )
        page = await context.new_page()

        # Navigate to a blank page first to have a stable context
        await page.goto("about:blank")

        print(f"Preparing to download from {url}...")
        
        # Start waiting for the download before clicking the link
        async with page.expect_download() as download_info:
            # Use page.evaluate to create and click a link to the URL
            await page.evaluate(f"() => {{ const a = document.createElement('a'); a.href = '{url}'; a.download = ''; a.click(); }}")

        download = await download_info.value
        print(f"Download started: {download.suggested_filename}")
        
        await download.save_as(out_path)
        print(f"Successfully downloaded {url} to {out_path}")
        
        return out_path
    except Exception as e:
        print(f"An error occurred while downloading {url}: {e}")
        # Attempt to screenshot for debugging
        if page:
            try:
                await page.screenshot(path="error_screenshot.png")
                print("Saved screenshot to error_screenshot.png")
            except Exception as se:
                print(f"Could not save screenshot: {se}")
        return None
    finally:
        if page and not page.is_closed():
            await page.close()
        if context:
            await context.close()


async def main_async():
    ensure_directory(PDFS_DIR)

    df = load_dataset("iknow-lab/open-materials-guide-2024")[
        "train"
    ].to_pandas()

    # Filter out records that are already in the download directory
    downloaded_files = [f.split(".")[0] for f in os.listdir(PDFS_DIR)]
    df = df[~df["id"].isin(downloaded_files)]

    if df.empty:
        print("No new records to process.")
        return

    async with async_playwright() as p:
        browser = await p.webkit.launch(headless=True)
        
        tasks = []
        with tqdm(total=len(df), desc="Downloading PDFs") as pbar:
            for _, row in df.iterrows():
                filename = f"{row['id']}.pdf"
                task = download_file_with_playwright(
                    browser, row["pdf_url"], PDFS_DIR, filename
                )
                tasks.append(task)

                if len(tasks) >= 8:  # BATCH_SIZE of 8
                    await asyncio.gather(*tasks)
                    pbar.update(len(tasks))
                    tasks = []
            
            if tasks:
                await asyncio.gather(*tasks)
                pbar.update(len(tasks))

        await browser.close()

    print("PDF download process finished.")


if __name__ == "__main__":
    asyncio.run(main_async())
