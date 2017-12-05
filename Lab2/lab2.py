

def clean_text(text: str) -> str:
    return text


def get_dataset(filename: str):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            text, label = line.rsplit(',', 1)
            if label not in {'spam', 'ham'}:
                raise ValueError('Unknown label: ' + label)
            data.append((text, label,))


