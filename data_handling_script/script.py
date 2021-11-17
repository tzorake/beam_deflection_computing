import re
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, help='filename', required=True)
    args = parser.parse_args()

    with open(args.f, 'r') as f:
        data = f.read()

    data = re.sub(r'\n\n', r'\n', data)
    data = re.sub(r'\t', r'', data)

    data = re.sub(r'\[\d\d:\d\d:\d\d\]\s\[\d+\]\s\[(.*)\]\n', r'', data)
    data = re.sub(r'\[\d\d:\d\d:\d\d\]\s\[\d+\]\s\[(.*)\]\n', r'', data)
    data = re.sub(r'\[\d\d:\d\d:\d\d\]\s\[uw\]\s\[(.*)\]\n', r'', data)
    data = re.sub(r'\[\d\d:\d\d:\d\d\]\s\[\d+\]\s\w+.*\n', r'', data)
    data = re.sub(r'\[\d\d:\d\d:\d\d\]\s\[\w\]\s', r'', data)
    data = re.sub(r'\[|\]', r'', data)

    data = re.split(r'\n', data)

    with open('out.txt', 'a') as f:
        f.truncate()
        for i in range(len(data)):
            f.write(f'{data[i]}\n')

if __name__ == '__main__':
    main()