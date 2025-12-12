"""
Quick test script to verify hate speech detection is working
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

print("✓ Vocabulary loaded:", len(vocab.word2idx), "words")

model = BiLSTMHateSpeechClassifier(
    vocab_size=len(vocab.word2idx),
    embedding_dim=128,
    hidden_dim=128,
    num_layers=2,
    dropout=0.2
)

checkpoint = torch.load('best_bilstm_model.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✓ Model loaded successfully")

def detect_hate_speech(text):
    """Detect hate speech in text"""
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
    
    threshold = 0.8
    is_hate = probability > threshold
    
    return is_hate, probability

# Test cases
test_cases = [
    ("Hello everyone! How are you today?", False),
    ("I hate you and all your kind", True),
    ("This is a nice day", False),
    ("You are stupid and worthless", True),
    ("I love this community!", False),
    ("Kill all those people", True),
]

print("\n" + "="*80)
print("TESTING HATE SPEECH DETECTION")
print("="*80)

all_passed = True
for text, expected_hate in test_cases:
    is_hate, probability = detect_hate_speech(text)
    status = "✓ PASS" if (is_hate == expected_hate) else "✗ FAIL"
    
    if is_hate != expected_hate:
        all_passed = False
    
    print(f"\n{status}")
    print(f"Text: {text}")
    print(f"Expected: {'HATE' if expected_hate else 'SAFE'}")
    print(f"Detected: {'HATE' if is_hate else 'SAFE'} (confidence: {probability*100:.1f}%)")

print("\n" + "="*80)
if all_passed:
    print("✓ ALL TESTS PASSED - Hate speech detection is working correctly!")
else:
    print("⚠ SOME TESTS FAILED - Check the results above")
print("="*80)
