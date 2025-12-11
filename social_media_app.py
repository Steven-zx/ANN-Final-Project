"""
Facebook-Style Social Media App with Hate Speech Detection
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import torch
from datetime import datetime
from rnn_model import BiLSTMHateSpeechClassifier
from text_preprocessing import TextPreprocessor, Vocabulary, pad_sequences


class SocialMediaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Social Media - Hate Speech Protected")
        self.root.geometry("800x700")
        self.root.configure(bg='#F0F2F5')
        
        # Load model
        self.device = torch.device('cpu')
        self.vocab = Vocabulary.load('vocabulary.pkl')
        self.preprocessor = TextPreprocessor(
            lowercase=True,
            remove_urls=True,
            remove_mentions=True,
            remove_hashtags=False,
            remove_punctuation=False
        )
        
        # Initialize model
        self.model = BiLSTMHateSpeechClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=128,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3,
            pad_idx=self.vocab.word2idx[self.vocab.PAD_TOKEN]
        ).to(self.device)
        
        checkpoint = torch.load('best_bilstm_model.pt', map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Store posts
        self.posts = []
        
        # Create UI
        self.create_header()
        self.create_post_button()
        self.create_feed()
        
    def create_header(self):
        """Create header with app title"""
        header = tk.Frame(self.root, bg='#1877F2', height=60)
        header.pack(fill='x', side='top')
        header.pack_propagate(False)
        
        title = tk.Label(
            header,
            text="📱 Social Media",
            font=('Segoe UI', 18, 'bold'),
            bg='#1877F2',
            fg='white'
        )
        title.pack(side='left', padx=20, pady=10)
        
        subtitle = tk.Label(
            header,
            text="Protected by AI Hate Speech Detection",
            font=('Segoe UI', 9),
            bg='#1877F2',
            fg='#E4E6EB'
        )
        subtitle.pack(side='left', padx=5, pady=10)
        
    def create_post_button(self):
        """Create 'What's on your mind?' button"""
        post_btn_frame = tk.Frame(self.root, bg='white', height=80)
        post_btn_frame.pack(fill='x', pady=10, padx=20)
        post_btn_frame.pack_propagate(False)
        
        # Profile icon
        profile_label = tk.Label(
            post_btn_frame,
            text="👤",
            font=('Segoe UI', 24),
            bg='white'
        )
        profile_label.pack(side='left', padx=15, pady=15)
        
        # Create post button
        post_btn = tk.Button(
            post_btn_frame,
            text="What's on your mind?",
            font=('Segoe UI', 11),
            bg='#F0F2F5',
            fg='#65676B',
            relief='flat',
            cursor='hand2',
            anchor='w',
            command=self.open_create_post_dialog
        )
        post_btn.pack(side='left', fill='both', expand=True, padx=(0, 15), pady=15)
        
    def create_feed(self):
        """Create scrollable feed area"""
        # Feed container
        feed_container = tk.Frame(self.root, bg='#F0F2F5')
        feed_container.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        # Canvas with scrollbar
        canvas = tk.Canvas(feed_container, bg='#F0F2F5', highlightthickness=0)
        scrollbar = ttk.Scrollbar(feed_container, orient='vertical', command=canvas.yview)
        
        self.feed_frame = tk.Frame(canvas, bg='#F0F2F5')
        
        self.feed_frame.bind(
            '<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all'))
        )
        
        canvas.create_window((0, 0), window=self.feed_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Welcome message
        self.show_welcome_message()
        
    def show_welcome_message(self):
        """Show welcome message when no posts"""
        welcome_frame = tk.Frame(self.feed_frame, bg='white', relief='solid', borderwidth=1)
        welcome_frame.pack(fill='x', pady=5)
        
        welcome_text = tk.Label(
            welcome_frame,
            text="👋 Welcome! Share your thoughts with the community.\n\nOur AI will keep the community safe by filtering hate speech.",
            font=('Segoe UI', 11),
            bg='white',
            fg='#65676B',
            justify='center',
            pady=30
        )
        welcome_text.pack()
        
    def open_create_post_dialog(self):
        """Open create post dialog (Facebook-style)"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Create post")
        dialog.geometry("500x400")
        dialog.configure(bg='white')
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        # Header
        header_frame = tk.Frame(dialog, bg='white')
        header_frame.pack(fill='x', pady=(0, 10))
        
        title = tk.Label(
            header_frame,
            text="Create post",
            font=('Segoe UI', 16, 'bold'),
            bg='white'
        )
        title.pack(pady=15)
        
        # Separator
        separator = tk.Frame(dialog, height=1, bg='#CED0D4')
        separator.pack(fill='x')
        
        # User info
        user_frame = tk.Frame(dialog, bg='white')
        user_frame.pack(fill='x', pady=15, padx=20)
        
        user_icon = tk.Label(
            user_frame,
            text="👤",
            font=('Segoe UI', 20),
            bg='white'
        )
        user_icon.pack(side='left', padx=(0, 10))
        
        user_info = tk.Frame(user_frame, bg='white')
        user_info.pack(side='left')
        
        username = tk.Label(
            user_info,
            text="Reliablesoft",
            font=('Segoe UI', 11, 'bold'),
            bg='white'
        )
        username.pack(anchor='w')
        
        privacy = tk.Label(
            user_info,
            text="🌐 Public",
            font=('Segoe UI', 9),
            bg='white',
            fg='#65676B'
        )
        privacy.pack(anchor='w')
        
        # Text input area
        text_frame = tk.Frame(dialog, bg='white')
        text_frame.pack(fill='both', expand=True, padx=20, pady=(0, 10))
        
        text_input = scrolledtext.ScrolledText(
            text_frame,
            font=('Segoe UI', 12),
            wrap='word',
            relief='flat',
            bg='white',
            fg='#050505',
            insertbackground='#1877F2'
        )
        text_input.pack(fill='both', expand=True)
        text_input.focus()
        
        # Placeholder
        placeholder = "What's on your mind?"
        text_input.insert('1.0', placeholder)
        text_input.config(fg='#65676B')
        
        def on_focus_in(event):
            if text_input.get('1.0', 'end-1c') == placeholder:
                text_input.delete('1.0', 'end')
                text_input.config(fg='#050505')
        
        def on_focus_out(event):
            if not text_input.get('1.0', 'end-1c').strip():
                text_input.insert('1.0', placeholder)
                text_input.config(fg='#65676B')
        
        text_input.bind('<FocusIn>', on_focus_in)
        text_input.bind('<FocusOut>', on_focus_out)
        
        # Add to post section
        add_frame = tk.Frame(dialog, bg='white', relief='solid', borderwidth=1)
        add_frame.pack(fill='x', padx=20, pady=(0, 10))
        
        add_label = tk.Label(
            add_frame,
            text="Add to your post",
            font=('Segoe UI', 10),
            bg='white',
            fg='#050505'
        )
        add_label.pack(side='left', padx=10, pady=10)
        
        # Icons
        icons_frame = tk.Frame(add_frame, bg='white')
        icons_frame.pack(side='right', padx=10, pady=10)
        
        for icon in ['🖼️', '👤', '😊', '📍', '🎬']:
            icon_label = tk.Label(
                icons_frame,
                text=icon,
                font=('Segoe UI', 14),
                bg='white',
                cursor='hand2'
            )
            icon_label.pack(side='left', padx=5)
        
        # Post button
        post_btn = tk.Button(
            dialog,
            text="Post",
            font=('Segoe UI', 12, 'bold'),
            bg='#1877F2',
            fg='white',
            relief='flat',
            cursor='hand2',
            height=2,
            command=lambda: self.create_post(text_input.get('1.0', 'end-1c'), dialog, placeholder)
        )
        post_btn.pack(fill='x', padx=20, pady=(0, 20), ipady=8)
        
    def create_post(self, text, dialog, placeholder):
        """Process and create post after hate speech check"""
        # Get text
        content = text.strip()
        
        # Check if empty or placeholder
        if not content or content == placeholder:
            messagebox.showwarning("Empty Post", "Please write something before posting.")
            return
        
        # Check for hate speech
        is_hate, probability = self.detect_hate_speech(content)
        
        if is_hate:
            # Show violation dialog
            self.show_violation_dialog(probability)
        else:
            # Post successfully
            self.add_post_to_feed(content)
            dialog.destroy()
            self.show_success_dialog(probability)
    
    def detect_hate_speech(self, text):
        """Detect hate speech in text"""
        processed = self.preprocessor.preprocess(text)
        sequence = self.vocab.text_to_sequence(processed, max_length=100)
        
        if len(sequence) == 0:
            sequence = [self.vocab.word2idx[self.vocab.UNK_TOKEN]]
        
        padded_seq, length = pad_sequences([sequence], max_length=100)
        text_tensor = torch.LongTensor(padded_seq).to(self.device)
        length_tensor = torch.LongTensor(length).to(self.device)
        
        with torch.no_grad():
            output = self.model(text_tensor, length_tensor)
            probability = torch.sigmoid(output).item()
        
        threshold = 0.8
        is_hate = probability > threshold
        
        return is_hate, probability
    
    def show_violation_dialog(self, probability):
        """Show hate speech violation dialog"""
        violation = tk.Toplevel(self.root)
        violation.title("Community Standards Violation")
        violation.geometry("450x300")
        violation.configure(bg='white')
        violation.resizable(False, False)
        violation.transient(self.root)
        violation.grab_set()
        
        # Center dialog
        violation.update_idletasks()
        x = (violation.winfo_screenwidth() // 2) - (violation.winfo_width() // 2)
        y = (violation.winfo_screenheight() // 2) - (violation.winfo_height() // 2)
        violation.geometry(f"+{x}+{y}")
        
        # Warning icon
        icon_label = tk.Label(
            violation,
            text="⚠️",
            font=('Segoe UI', 48),
            bg='white'
        )
        icon_label.pack(pady=(30, 10))
        
        # Title
        title = tk.Label(
            violation,
            text="Post Blocked",
            font=('Segoe UI', 18, 'bold'),
            bg='white',
            fg='#D93025'
        )
        title.pack(pady=5)
        
        # Message
        message = tk.Label(
            violation,
            text=f"Your post goes against our Community Standards on hate speech.\n\n"
                 f"We don't allow content that attacks people based on their\n"
                 f"protected characteristics or promotes violence.\n\n"
                 f"Detection Confidence: {probability*100:.1f}%",
            font=('Segoe UI', 10),
            bg='white',
            fg='#65676B',
            justify='center'
        )
        message.pack(pady=10)
        
        # OK button
        ok_btn = tk.Button(
            violation,
            text="I Understand",
            font=('Segoe UI', 11, 'bold'),
            bg='#1877F2',
            fg='white',
            relief='flat',
            cursor='hand2',
            command=violation.destroy
        )
        ok_btn.pack(pady=(20, 30), ipadx=30, ipady=5)
    
    def show_success_dialog(self, probability):
        """Show successful post dialog"""
        success = tk.Toplevel(self.root)
        success.title("Post Published")
        success.geometry("400x250")
        success.configure(bg='white')
        success.resizable(False, False)
        success.transient(self.root)
        success.grab_set()
        
        # Center dialog
        success.update_idletasks()
        x = (success.winfo_screenwidth() // 2) - (success.winfo_width() // 2)
        y = (success.winfo_screenheight() // 2) - (success.winfo_height() // 2)
        success.geometry(f"+{x}+{y}")
        
        # Success icon
        icon_label = tk.Label(
            success,
            text="✅",
            font=('Segoe UI', 48),
            bg='white'
        )
        icon_label.pack(pady=(30, 10))
        
        # Title
        title = tk.Label(
            success,
            text="Post Published Successfully!",
            font=('Segoe UI', 16, 'bold'),
            bg='white',
            fg='#00A400'
        )
        title.pack(pady=5)
        
        # Message
        confidence = (1 - probability) * 100
        message = tk.Label(
            success,
            text=f"Your post is clean and safe for the community.\n\n"
                 f"Safety Score: {confidence:.1f}%",
            font=('Segoe UI', 10),
            bg='white',
            fg='#65676B',
            justify='center'
        )
        message.pack(pady=10)
        
        # Close button
        close_btn = tk.Button(
            success,
            text="Close",
            font=('Segoe UI', 11, 'bold'),
            bg='#1877F2',
            fg='white',
            relief='flat',
            cursor='hand2',
            command=success.destroy
        )
        close_btn.pack(pady=(15, 30), ipadx=30, ipady=5)
        
        # Auto close after 2 seconds
        success.after(2000, success.destroy)
    
    def add_post_to_feed(self, content):
        """Add post to feed"""
        # Clear welcome message if first post
        if len(self.posts) == 0:
            for widget in self.feed_frame.winfo_children():
                widget.destroy()
        
        # Add to posts list
        timestamp = datetime.now()
        self.posts.insert(0, {'content': content, 'timestamp': timestamp})
        
        # Create post card
        post_frame = tk.Frame(self.feed_frame, bg='white', relief='solid', borderwidth=1)
        post_frame.pack(fill='x', pady=5)
        
        # Post header
        header = tk.Frame(post_frame, bg='white')
        header.pack(fill='x', padx=15, pady=10)
        
        user_icon = tk.Label(
            header,
            text="👤",
            font=('Segoe UI', 20),
            bg='white'
        )
        user_icon.pack(side='left', padx=(0, 10))
        
        user_info = tk.Frame(header, bg='white')
        user_info.pack(side='left')
        
        username = tk.Label(
            user_info,
            text="Reliablesoft",
            font=('Segoe UI', 11, 'bold'),
            bg='white'
        )
        username.pack(anchor='w')
        
        # Time format
        time_str = self.format_timestamp(timestamp)
        time_label = tk.Label(
            user_info,
            text=f"{time_str} · 🌐",
            font=('Segoe UI', 9),
            bg='white',
            fg='#65676B'
        )
        time_label.pack(anchor='w')
        
        # Post content
        content_label = tk.Label(
            post_frame,
            text=content,
            font=('Segoe UI', 12),
            bg='white',
            fg='#050505',
            justify='left',
            wraplength=700,
            anchor='w'
        )
        content_label.pack(fill='x', padx=15, pady=(0, 15))
        
        # Separator
        separator = tk.Frame(post_frame, height=1, bg='#CED0D4')
        separator.pack(fill='x', padx=15)
        
        # Actions (Like, Comment, Share)
        actions_frame = tk.Frame(post_frame, bg='white')
        actions_frame.pack(fill='x', pady=8)
        
        actions = [
            ('👍', 'Like'),
            ('💬', 'Comment'),
            ('↗️', 'Share')
        ]
        
        for icon, text in actions:
            action_btn = tk.Label(
                actions_frame,
                text=f"{icon} {text}",
                font=('Segoe UI', 10),
                bg='white',
                fg='#65676B',
                cursor='hand2'
            )
            action_btn.pack(side='left', expand=True)
    
    def format_timestamp(self, timestamp):
        """Format timestamp for display"""
        now = datetime.now()
        diff = now - timestamp
        
        if diff.seconds < 60:
            return "Just now"
        elif diff.seconds < 3600:
            minutes = diff.seconds // 60
            return f"{minutes}m"
        elif diff.seconds < 86400:
            hours = diff.seconds // 3600
            return f"{hours}h"
        else:
            return timestamp.strftime("%B %d")


def main():
    root = tk.Tk()
    app = SocialMediaApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
