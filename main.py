from perplexity import detect


def load_jsonl(filename):
    import json
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    data = load_jsonl("solutions.jsonl")

    for item in data:
        code = item["code"]
        label = item["label"]

        result = detect(code)

        print(f"Label: {label}")
        print(f"Result: {result}")

if __name__ == "__main__":
    main()
