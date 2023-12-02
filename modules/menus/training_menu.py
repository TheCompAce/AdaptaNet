from modules.llms.llm import LLM
from modules.AdaptaNetModel import AdaptaNetModel
from modules.Vectors import EmbeddingTrain  # Import the training function

def main():
    while True:
        print("[1] Train Embeddings")
        print("[2] Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            train_embeddings()
        elif choice == '2':
            print("Exiting Testing Menu.")
            break
        else:
            print("Invalid choice, please try again.")

def train_embeddings():
    settings_file = 'settings.json'
    embeddings_model = LLM(model_name='jinaai/jina-embeddings-v2-base-en', use_causal_pretrained=False, trust_remote_code=True) # Initialize your embeddings model here
    EmbeddingTrain.train_adaptanet_model(num_epochs=5, settings_file=settings_file, input_dim=128, hidden_dim=256, output_dim=128, num_heads=4, num_layers=2, embeddings_model=embeddings_model)
    print("Training complete.")
