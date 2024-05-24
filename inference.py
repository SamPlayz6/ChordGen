from train import LSTMModel, torch, tokenize_and_pad, device, chord_token_to_id
from train import save_model_path, melody_tokenizer, input_size, embedding_dim, hidden_size, output_size, num_layers, dropout, max_len
print("Starting inference...")

def load_model(path, input_size, embedding_dim, hidden_size, output_size, num_layers, dropout):
    model = LSTMModel(input_size, embedding_dim, hidden_size, output_size, num_layers, dropout)
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    return model


#To make predictions with a loaded model, you can define an inference function that prepares the input, feeds it to the model, and processes the output:
def predict(model, tokenizer, input_sequence, max_len):
    # Tokenize and pad the input sequence
    tokenized_input = tokenize_and_pad([input_sequence], tokenizer, max_len)
    input_tensor = torch.tensor(tokenized_input, dtype=torch.long).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        predicted_index = output.argmax(1).item()  # Get the predicted class index

    # Convert index to chord (reverse lookup in your token-to-id map)
    id_to_token = {v: k for k, v in chord_token_to_id.items()}  # assuming chord_token_to_id is accessible
    predicted_chord = id_to_token[predicted_index]
    return predicted_chord

# Load the model
loaded_model = load_model(save_model_path, input_size, embedding_dim, hidden_size, output_size, num_layers, dropout)

# Example input (modify according to actual input expected)
example_input_sequence = "A0/5,F1/5,G0/5,A0/5,F1/5,G0/5,A0/5,G0/5,D0/6,B0/5,A0/5,G0/5,D0/5,E0/5,G0/5,F1/5,C1/6,C1/6,C0/6,B0/5,C1/5,D0/5,E0/5,A0/5,A0/5,F1/5,G0/5,D0/5,D0/5,F1/5,A0/5,C1/6,E0/6,D0/6,D0/5,C1/5,D0/5,E0/6,E0/6,D0/6,B0/5,A0/5,F1/5,C0/6,C1/6,C0/6,B0/5,A0/5,G0/5,F1/5,D0/5,F1/5,E0/5,B0/5,F0/5,F1/5,G0/5,A0/5,F1/5,G0/5,A0/5,F1/5,G0/5,A0/5,G0/5,D0/6,B0/5,A0/5,G0/5,D0/5,E0/5,G0/5,F1/5,C1/6,C1/6,C0/6,B0/5,C1/5,D0/5,E0/5,D0/5,D0/5"

# Predict using the loaded model
predicted_chord = predict(loaded_model, melody_tokenizer, example_input_sequence, max_len)
print("Predicted Chord:", predicted_chord)