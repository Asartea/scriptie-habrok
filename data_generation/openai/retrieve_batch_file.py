from openai import OpenAI


def main():
    batch_file = input("Enter batch ID: ").strip()
    client = OpenAI()

    file_response = client.files.content(batch_file)
    print(f"Batch file content for {batch_file}:")
    for line in file_response.text.splitlines():
        print(line)


if __name__ == "__main__":
    main()
