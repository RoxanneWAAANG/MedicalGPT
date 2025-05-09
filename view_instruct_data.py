import json

INPUT_PATH = '/home/jack/Projects/yixin-llm/yixin-llm-data/instruct_dataset/mimic-cxr-5k/annotation.json'
OUTPUT_PATH = 'output.json'

def main():
    # Load the JSON
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Slice first five terms
    if isinstance(data, list):
        first_five = data[:5]
    elif isinstance(data, dict):
        # dict preserves insertion order (Python 3.7+)
        first_five = dict(list(data.items())[:1])
    else:
        raise TypeError(f"Unsupported JSON top-level type: {type(data)}")

    # Write out the slice
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(first_five, f, ensure_ascii=False, indent=2)

    print(f"Wrote first five terms to {OUTPUT_PATH!r}")

if __name__ == '__main__':
    main()