from importlib.metadata import version
import tiktoken

print("tikitoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

print(text)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)
print(tokenizer.decode(integers))

text2 = "Akwirw ier"
integers = tokenizer.encode(text2, allowed_special={"<|endoftext|>"})
print(integers)
print(tokenizer.decode(integers))
