import tiktoken

enc = tiktoken.get_encoding("gpt2")

import tiktoken

enc = tiktoken.get_encoding("gpt2")

if __name__ == "__main__":
    text = "I love Brando Code videos"
    token_id = enc.encode(text)

    print(token_id)


