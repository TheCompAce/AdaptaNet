import torch
from modules.llms.llm import LLM
from modules.AdaptaNetModel import AdaptaNetModel
from modules.Vectors.EmbeddingVectorHouse import EmbeddingVectorHouse
from modules.utils import get_prompt_from_settings  # Import the utility function

def train_adaptanet_model(num_epochs, settings_file, input_dim, hidden_dim, output_dim, num_heads, num_layers, embeddings_model):
    model = AdaptaNetModel(input_dim, hidden_dim, output_dim, num_heads, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    llm = LLM()

    for epoch in range(num_epochs):
        system_prompt = get_prompt_from_settings(settings_file, 'system')
        user_prompt = get_prompt_from_settings(settings_file, 'user')

        # Generate response using the prompts
        full_response = llm.generate_response(system_prompt, user_prompt)

        print(f"full_response = {full_response}")

        # Process the full response and system/user prompts
        training_vector_house = EmbeddingVectorHouse(full_response, embeddings_model)
        input_tensor = training_vector_house.to_tensor()

        target_vector_house = EmbeddingVectorHouse(system_prompt + user_prompt, embeddings_model)
        target_tensor = target_vector_house.to_tensor()

        output = model(input_tensor)

        loss = EmbeddingVectorHouse.loss_function(output, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Example usage
settings_file = 'settings.json'
embeddings_model = LLM(model_name='jinaai/jina-embeddings-v2-base-en', use_causal_pretrained=False, trust_remote_code=True) # Initialize your embeddings model here
train_adaptanet_model(num_epochs=5, settings_file=settings_file, input_dim=128, hidden_dim=256, output_dim=128, num_heads=4, num_layers=2, embeddings_model=embeddings_model)
