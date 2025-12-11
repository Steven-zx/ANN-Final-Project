"""
Quick test of GUI predictions with new threshold
"""

import torch
from rnn_model import BiLSTMHateSpeechClassifier
from text_preprocessing import TextPreprocessor, Vocabulary, pad_sequences

# Load model
device = torch.device('cpu')
vocab = Vocabulary.load('vocabulary.pkl')

preprocessor = TextPreprocessor(
    lowercase=True,
    remove_urls=True,
    remove_mentions=True,
    remove_hashtags=False,
    remove_punctuation=False
)

model = BiLSTMHateSpeechClassifier(
    vocab_size=len(vocab),
    embedding_dim=128,
    hidden_dim=128,
    num_layers=2,
    dropout=0.3,
    pad_idx=vocab.word2idx[vocab.PAD_TOKEN]
).to(device)

checkpoint = torch.load('best_bilstm_model.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("="*80)
print("TESTING WITH THRESHOLD = 0.8 (GUI Setting)")
print("="*80)

test_cases = [
    ("you are the best thing in the world", False),
    ("thank you so much for your help", False),
    ("i love this product", False),
    ("have a great day", False),
    ("you're amazing and talented", False),
    ("you're a stupid idiot", True),
    ("i hate you so much", True),
    ("tang ina mo gago ka", True),
]

threshold = 0.8
correct = 0

for text, expected_hate in test_cases:
    processed = preprocessor.preprocess(text)
    sequence = vocab.text_to_sequence(processed, max_length=100)
    
    if len(sequence) == 0:
        sequence = [vocab.word2idx[vocab.UNK_TOKEN]]
    
    padded_seq, length = pad_sequences([sequence], max_length=100)
    text_tensor = torch.LongTensor(padded_seq).to(device)
    length_tensor = torch.LongTensor(length).to(device)
    
    with torch.no_grad():
        output = model(text_tensor, length_tensor)
        probability = torch.sigmoid(output).item()
    
    is_hate = probability > threshold
    confidence = probability if is_hate else (1 - probability)
    
    result = "⚠️ HATE" if is_hate else "✅ SAFE"
    expected = "⚠️ HATE" if expected_hate else "✅ SAFE"
    status = "✅" if (is_hate == expected_hate) else "❌"
    
    if is_hate == expected_hate:
        correct += 1
    
    print(f"{status} {result} (Confidence: {confidence*100:.1f}%) | Expected: {expected}")
    print(f"   Text: \"{text}\"")
    print(f"   Raw probability: {probability:.3f}")
    print()

accuracy = correct / len(test_cases) * 100
print("="*80)
print(f"ACCURACY: {correct}/{len(test_cases)} ({accuracy:.1f}%)")
print("="*80)
print("\n✅ GUI has been updated with threshold = 0.8")
print("Run: python gui_app.py")
