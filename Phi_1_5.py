from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model
# model_name = "microsoft/phi-1_5"
model_name = "./models"
device = 'cuda:1' if torch.cuda.is_available else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
model.to(device)

# Define the context with triple quotes
context = """
"""

prompt = '''```python
def print_prime(n):
    """
    print all prime numbers between 1 and n.
    """'''


max_length = 1000
bos_token_id = tokenizer.bos_token_id
context_encoding = tokenizer(context, return_tensors="pt").to(device)
prompt_encoding = tokenizer(prompt, return_tensors="pt").to(device)
inputs = torch.cat([context_encoding["input_ids"], prompt_encoding["input_ids"]], dim=1)
outputs = model.generate(input_ids=inputs, max_length=max_length, bos_token_id=bos_token_id)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)
