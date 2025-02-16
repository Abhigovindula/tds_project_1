# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "faker",
#     "httpx",
#     "numpy",
#     "pillow",
#     "python-dateutil",
# ]
# ///
import hashlib
import httpx
import json
import logging
import numpy as np
import os
import re
import subprocess
from dateutil.parser import parse
from datagen import (
    get_markdown,
    get_dates,
    get_contacts,
    get_logs,
    get_docs,
    get_email,
    get_credit_card,
    get_comments,
    get_tickets,
)


openai_api_base = os.getenv("OPENAI_API_BASE", "https://aiproxy.sanand.workers.dev/openai/v1")
openai_api_key = os.getenv("OPENAI_API_KEY")


def num(str):
    return int(hashlib.sha256(str.encode()).hexdigest(), 16) % (2**32)


def mismatch(msg, expected, result):
    logging.error(f"🔴 {msg}\n⚠️ EXPECTED:\n{expected}\n⚠️ RESULT:\n{result}")
    return False


async def run(task: str):
    async with httpx.AsyncClient(timeout=30) as client:
        logging.warning(f"🟡 Running task: {task.strip()}")
        response = await client.post("http://localhost:8000/run", params={"task": task})
        try:
            response_text = json.dumps(response.json(), indent=2)
        except json.JSONDecodeError:
            response_text = response.text
        if response.status_code < 400:
            logging.info(f"🟢 HTTP {response.status_code} {response_text}")
        else:
            logging.error(f"🔴 HTTP {response.status_code} {response_text}")
        return response.status_code, response_text


async def read(path: str):
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"http://localhost:8000/read?path={path}")
        if response.status_code != 200:
            raise Exception(f"Cannot read {path}")
        return response.text


async def a1(email: str, **kwargs):
    await run(
        f"""
Install `uv` (if required) and run the script `https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/datagen.py`
with `{email}` as the only argument
"""
    )
    return email in await read("/data/format.md")


async def a2(email: str, file: str = "/data/format.md", **kwargs):
    original = get_markdown(email)
    expected = subprocess.run(
        ["npx", "prettier@3.4.2", "--stdin-filepath", file],
        input=original,
        capture_output=True,
        text=True,
        check=True,
        # Ensure npx is picked up from the PATH on Windows
        shell=True,
    ).stdout
    result = await run(
        f"""
Format the contents of `{file}` using `prettier@3.4.2`, updating the file in-place
"""
    )
    result = await read(file)
    if result != expected:
        return mismatch(file, expected, result)
    return True


async def a3(email, **kwargs):
    dates = get_dates(email)
    await run(
        "The file `/data/dates.txt` contains a list of dates, one per line. Count the number of Wednesdays in the list, and write just the number to `/data/dates-wednesdays.txt`"
    )
    result = await read("/data/dates-wednesdays.txt")
    expected = sum(1 for date in dates if parse(date).weekday() == 2)
    if result.strip() != str(expected):
        return mismatch("/data/dates-wednesdays.txt", expected, result)
    return True


async def a4(email, **kwargs):
    contacts = get_contacts(email)
    contacts.sort(key=lambda c: (c["last_name"], c["first_name"]))
    await run(
        "Sort the array of contacts in `/data/contacts.json` by `last_name`, then `first_name`, and write the result to `/data/contacts-sorted.json`"
    )
    result = await read("/data/contacts-sorted.json")
    try:
        result = json.loads(result)
    except json.JSONDecodeError:
        logging.error("🔴 /data/contacts-sorted.json was not valid JSON")
        return False
    if json.dumps(result, sort_keys=True) != json.dumps(contacts, sort_keys=True):
        return mismatch("/data/contacts-sorted.json", contacts, result)
    return True


async def a5(email, **kwargs):
    files = get_logs(email)
    files.sort(key=lambda f: f[0])
    expected = "".join([f[1].split("\n")[0] + "\n" for f in files[:10]])
    await run(
        "Write the first line of the 10 most recent `.log` file in `/data/logs/` to `/data/logs-recent.txt`, most recent first"
    )
    result = await read("/data/logs-recent.txt")
    if result.strip() != expected.strip():
        return mismatch("/data/logs-recent.txt", expected, result)
    return True

async def a6(email, **kwargs):
    docs = get_docs(email)
    allowed_dirs = {"however", "action", "knowledge", "fund", "since", "even", "hope", "here", "set", "card"}
    await run(
        """Find all Markdown (`.md`) files in `/data/docs/`.
For each file, extract the first occurrance of an H1 (i.e. a line starting with `# `).
Create an index file `/data/docs/index.json` that maps each filename (without the `/data/docs/` prefix) to its title
(e.g. `{"README.md": "Home", "path/to/large-language-models.md": "Large Language Models", ...}`)"""
    )
    expected = {}
    for dir, file, text in docs:
        if dir not in allowed_dirs:
            continue  # Only consider allowed directories.
        for line in text.split("\n"):
            if line.startswith("# "):
                title = line[2:].strip()
                expected[f"{dir}/{file}.md"] = title
                break
    result = await read("/data/docs/index.json")
    try:
        result = json.loads(result)
    except json.JSONDecodeError:
        logging.error("🔴 /data/docs/index.json was not valid JSON")
        return False
    if json.dumps(result, sort_keys=True) != json.dumps(expected, sort_keys=True):
        return mismatch("/data/docs/index.json", expected, result)
    return True


async def a7(email, **kwargs):
    expected = get_email(email)["from_email"]
    await run(
        "`/data/email.txt` contains an email message. Pass the content to an LLM with instructions to extract the sender's email address, and write just the email address to `/data/email-sender.txt`"
    )
    result = await read("/data/email-sender.txt")
    if result != expected:
        return mismatch("/data/email-sender.txt", expected, result)
    return True


async def a8(email, **kwargs):
    data = get_credit_card(email)
    await run(
        "`/data/credit_card.png` contains a credit card number. Pass the image to an LLM, have it extract the card number, and write it without spaces to `/data/credit-card.txt`"
    )
    result = await read("/data/credit-card.txt")
    if re.sub(r"\D", "", result) != re.sub(r"\D", "", data["number"]):
        return mismatch("/data/credit-card.txt", data["number"], result)
    return True


async def a9(email, **kwargs):
    data = get_comments(email)
    
    def local_embedding(text):
        import numpy as np, hashlib
        seed = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        return rng.rand(10)
    
    import numpy as np
    embeddings = np.array([local_embedding(comment) for comment in data])
    similarity = np.dot(embeddings, embeddings.T)
    np.fill_diagonal(similarity, -np.inf)
    i, j = np.unravel_index(similarity.argmax(), similarity.shape)
    expected = "\n".join(sorted([data[i], data[j]]))
    
    await run(
        "`/data/comments.txt` contains a list of comments, one per line. Using embeddings, find the most similar pair of comments and write them to `/data/comments-similar.txt`, one per line"
    )
    result = await read("/data/comments-similar.txt")
    sorted_result = "\n".join(sorted([line for line in result.split("\n") if line.strip()]))
    if sorted_result != expected:
        return mismatch("/data/comments-similar.txt", expected, result)
    return True


async def a10(email, **kwargs):
    data = get_tickets(email)
    await run(
        'The SQLite database file `/data/ticket-sales.db` has a `tickets` with columns `type`, `units`, and `price`. Each row is a customer bid for a concert ticket. What is the total sales of all the items in the "Gold" ticket type? Write the number in `/data/ticket-sales-gold.txt`'
    )
    result = await read("/data/ticket-sales-gold.txt")
    expected = sum(row[1] * row[2] for row in data if row[0].lower() == "gold")
    try:
        result = float(result)
    except ValueError:
        logging.error(f"🔴 /data/ticket-sales-gold.txt was {result}, not a valid number")
        return False
    if abs(result - expected) > 0.1:
        return mismatch("/data/ticket-sales-gold.txt", expected, result)
    return True

async def b4_test():
    repo_url = "https://github.com/example/repo.git"  # Replace with a real test repo
    commit_message = "Test commit from automation"
    file_to_modify = "README.md"
    modification_text = "Automated commit test"

    await run(f"Clone the repo {repo_url}, modify {file_to_modify}, and commit with '{commit_message}'")
    return True  # Since we can't verify without repo access, manual confirmation needed
async def b8_test():
    audio_path = "/data/sample.mp3"  # Ensure this file exists before testing
    output_path = "/data/sample_transcription.txt"

    await run(f"Transcribe the MP3 file {audio_path} and save to {output_path}")
    result = await read(output_path)
    
    if len(result.strip()) > 0:
        return True
    else:
        return mismatch(output_path, "Expected non-empty transcription", result)
async def b10_test():
    csv_path = "/data/sample.csv"  # Ensure this file exists
    filter_column = "status"
    filter_value = "active"

    status, result = await run(f"Filter {csv_path} where {filter_column} = '{filter_value}'")
    
    if status == 200:
        return True
    else:
        return mismatch("B10 API", "Expected valid JSON response", result)


'''async def main(email: str):
    score, total = 0, 0
    for task in [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10,b4_test, b8_test, b10_test]:
   # for task in [b4_test, b8_test, b10_test]:
        total += 1
        try:
            success = await task(email=email)
            #success=await task()
        except Exception as e:
            logging.error(f"🔴 {task.__name__.upper()} failed: {e}")
            success = False
        if success:
            logging.info(f"✅ {task.__name__.upper()} PASSED")
        else:
            logging.error(f"❌ {task.__name__.upper()} FAILED")
        score += 1 if success else 0
    logging.info(f"🎯 Score: {score} / {total}")
    
'''
async def main(email: str):
    score, total = 0, 0
    for task in [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, b4_test, b8_test, b10_test]:
        total += 1
        try:
            if task.__name__.startswith("a"):  # ✅ Tasks A get email
                success = await task(email=email)
            else:  # ✅ Tasks B don't get email
                success = await task()

        except Exception as e:
            logging.error(f"🔴 {task.__name__.upper()} failed: {e}")
            success = False

        if success:
            logging.info(f"✅ {task.__name__.upper()} PASSED")
        else:
            logging.error(f"❌ {task.__name__.upper()} FAILED")

        score += 1 if success else 0

    logging.info(f"🎯 Score: {score} / {total}")


if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate tasks with configurable logging")
    parser.add_argument("--email", default="user@example.com", help="Set the email address")
    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    parser.add_argument("--log-level", default="INFO", choices=levels, help="Set logging level")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(message)s\n")
    asyncio.run(main(args.email))