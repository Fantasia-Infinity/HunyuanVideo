import os

example_input_file = "/projects/prjs1914/input/qwen_describe/0001_fw.txt"

def index_to_file_path(index, input_dir="/projects/prjs1914/input/qwen_describe"):
    return os.path.join(input_dir, f"{index:04d}_fw.txt")


max_len = 0
lens = []
for idx in range(1, 2180):
    file_path = index_to_file_path(idx)
    if not os.path.exists(file_path):
        print(f"[Warning] {file_path} not found, skipping.")
        continue
    with open(file_path, "r") as f:
        content = f.read()
        #the number of tokens is approximated by the number of whitespace-separated words
        num_tokens = len(content.split())
        lens.append(num_tokens)
        if num_tokens > max_len:
            max_len = num_tokens
            print(f"New max token count {max_len} found in {file_path}")    

print(f"Maximum token count across all files: {max_len}")   


draw_histogram = True
if draw_histogram:
    import matplotlib.pyplot as plt

    plt.hist(lens, bins=50, edgecolor='black')
    plt.title('Distribution of Token Counts in Text Files')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    #save the picture
    plt.hist(lens, bins=50, edgecolor='black')
    plt.title('Distribution of Token Counts in Text Files')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig("/projects/prjs1914/output/hunyuan_text_embeddings/token_count_histogram.png")