import time
from openai import OpenAI
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python watcher.py <batch_id>")
        sys.exit(1)
    batch_id = sys.argv[1].strip()
    client = OpenAI()

    terminal_states = {"completed", "failed", "expired", "cancelled"}

    while True:
        batch = client.batches.retrieve(batch_id)

        print(f"[{batch.id}] status = {batch.status}")

        if batch.status in terminal_states:
            print("Done!")
            print(batch)
            break

        time.sleep(30)


if __name__ == "__main__":
    main()
