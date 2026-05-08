from perplexity import detect


def load_jsonl(filename):
    import json
    data = []
    with open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    data = load_jsonl("data.jsonl")

    for item in data:
        code = item["code"]
        label = item["label"]

        score = detect(code)

        print(f"Label: {label}, Score: {score:.4f}")

if __name__ == "__main__":
    main()
