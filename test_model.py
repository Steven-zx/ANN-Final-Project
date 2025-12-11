"""
Test the already trained model
"""

import torch
from rnn_model import BiLSTMHateSpeechClassifier, count_parameters
from text_preprocessing import Vocabulary

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load vocabulary
print("\nLoading vocabulary...")
vocab = Vocabulary.load('vocabulary.pkl')

# Initialize model
print("\nInitializing model...")
model = BiLSTMHateSpeechClassifier(
    vocab_size=len(vocab),
    embedding_dim=128,
    hidden_dim=128,
    num_layers=2,
    dropout=0.3,
    pad_idx=vocab.word2idx[vocab.PAD_TOKEN]
).to(device)

print(f"Model parameters: {count_parameters(model):,}")

# Load trained weights
print("\nLoading trained model...")
checkpoint = torch.load('best_bilstm_model.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded from epoch {checkpoint['epoch']+1}")
print(f"✓ Validation accuracy: {checkpoint['val_acc']:.4f}")
print(f"✓ Validation F1: {checkpoint['val_f1']:.4f}")

print("\n" + "="*70)
print("MODEL READY FOR USE!")
print("="*70)
print("\nYou can now:")
print("1. Run the GUI: python gui_app.py")
print("2. Or test with sample text below:")
print("="*70)

# Test with sample texts
from text_preprocessing import TextPreprocessor, pad_sequences

preprocessor = TextPreprocessor(
    lowercase=True,
    remove_urls=True,
    remove_mentions=True,
    remove_hashtags=False,
    remove_punctuation=False
)

test_texts = [
    "Salamat sa suporta! You're the best!",
    "Tang ina mo! Gago ka!",
    "This is a normal message without hate.",
    "You're such an idiot and a piece of trash!"
]

print("\nTesting with sample texts:")
print("-"*70)

for text in test_texts:
    # Preprocess
    processed = preprocessor.preprocess(text)
    sequence = vocab.text_to_sequence(processed, max_length=100)
    
    if len(sequence) == 0:
        sequence = [vocab.word2idx[vocab.UNK_TOKEN]]
    
    padded_seq, length = pad_sequences([sequence], max_length=100)
    
    # Convert to tensors
    text_tensor = torch.LongTensor(padded_seq).to(device)
    length_tensor = torch.LongTensor(length).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(text_tensor, length_tensor)
        probability = torch.sigmoid(output).item()
    
    is_hate = probability > 0.5
    confidence = probability if is_hate else (1 - probability)
    
    result = "⚠️ HATE SPEECH" if is_hate else "✅ NO HATE"
    
    print(f"\nText: {text}")
    print(f"Result: {result} (Confidence: {confidence*100:.1f}%)")

print("\n" + "="*70)
