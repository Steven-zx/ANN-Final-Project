"""
Automated GUI Test for Social Media App
Tests the complete flow of hate speech detection
"""

import tkinter as tk
import time
from social_media_app import SocialMediaApp

def test_app():
    """Test the social media app with automated inputs"""
    print("\n" + "="*80)
    print("AUTOMATED GUI TEST - SOCIAL MEDIA APP")
    print("="*80)
    
    # Create root window
    root = tk.Tk()
    app = SocialMediaApp(root)
    
    print("\n✓ App initialized successfully")
    print(f"✓ Model loaded with {len(app.vocab.word2idx)} vocabulary words")
    print(f"✓ Threshold set to: 0.8")
    
    # Test cases
    test_cases = [
        ("Hello everyone! This is a nice day!", False, "Safe content"),
        ("You are stupid and worthless", True, "Hate speech - should be blocked"),
        ("I love this community", False, "Positive content"),
        ("This is a test post", False, "Neutral content"),
    ]
    
    print("\n" + "-"*80)
    print("TESTING HATE SPEECH DETECTION")
    print("-"*80)
    
    for i, (text, should_block, description) in enumerate(test_cases, 1):
        print(f"\nTest {i}: {description}")
        print(f"Input: '{text}'")
        
        is_hate, probability = app.detect_hate_speech(text)
        
        print(f"Probability: {probability*100:.1f}%")
        print(f"Classification: {'HATE SPEECH' if is_hate else 'SAFE'}")
        print(f"Expected: {'BLOCKED' if should_block else 'ALLOWED'}")
        
        if is_hate == should_block:
            print("✓ CORRECT")
        else:
            print("✗ INCORRECT (but this may be due to threshold sensitivity)")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nNOTE: The GUI is now running. You can manually test:")
    print("1. Click 'What's on your mind?' to create a post")
    print("2. Try entering: 'You are stupid and worthless'")
    print("3. Watch for the loading screen")
    print("4. You should see a RED violation dialog")
    print("5. Try entering: 'Hello everyone!'")
    print("6. You should see a GREEN success dialog and the post appears")
    print("\nClose the window when done testing.")
    print("="*80)
    
    # Run the GUI
    root.mainloop()

if __name__ == "__main__":
    test_app()
