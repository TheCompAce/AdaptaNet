from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_response(system_input, user_input, model_name='Intel/neural-chat-7b-v3-1', max_length=1000):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prompt = f"### System:\n{system_input}\n### User:\n{user_input}\n### Assistant:\n"
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response.split("### Assistant:\n")[-1]

# Example usage
system_input = "You are a math expert assistant. Your mission is to help users understand and solve various math problems."
user_input = "calculate 100 + 520 + 60"
response = generate_response(system_input, user_input)
print(response)
