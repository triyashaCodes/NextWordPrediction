import requests
import os
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
import numpy as np

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SherlockHolmesDataAnalyzer:
    def __init__(self):
        self.data_url = "https://www.gutenberg.org/files/1661/1661-0.txt"
        self.local_file = "sherlock_holmes_gutenberg.txt"
        self.cleaned_file = "sherlock_holmes_cleaned.txt"
        
    def download_data(self):
        """Download Sherlock Holmes data from Project Gutenberg"""
        print("=" * 60)
        print("DOWNLOADING SHERLOCK HOLMES DATASET")
        print("=" * 60)
        
        if os.path.exists(self.local_file):
            print(f"Data already exists: {self.local_file}")
            return True
            
        try:
            print(f"Downloading from: {self.data_url}")
            response = requests.get(self.data_url)
            response.raise_for_status()
            
            with open(self.local_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"Data downloaded successfully: {self.local_file}")
            print(f"File size: {os.path.getsize(self.local_file) / 1024:.2f} KB")
            return True
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            return False
    
    def clean_data(self):
        """Clean the downloaded data by removing headers, footers, and formatting"""
        print("\n" + "=" * 60)
        print("CLEANING DATASET")
        print("=" * 60)
        
        try:
            with open(self.local_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove Project Gutenberg header and footer
            start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
            end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
            
            start_idx = content.find(start_marker)
            end_idx = content.find(end_marker)
            
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx]
            
            # Remove extra whitespace and normalize
            lines = content.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith('***') and len(line) > 10:
                    # Remove page numbers and other artifacts
                    line = re.sub(r'^\d+\s*$', '', line)  # Remove standalone numbers
                    line = re.sub(r'\[.*?\]', '', line)   # Remove bracketed text
                    line = re.sub(r'^\s*[A-Z\s]+\s*$', '', line)  # Remove all caps headers
                    if line.strip():
                        cleaned_lines.append(line)
            
            # Write cleaned data
            with open(self.cleaned_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(cleaned_lines))
            
            print(f"Data cleaned and saved: {self.cleaned_file}")
            print(f"Original lines: {len(lines)}")
            print(f"Cleaned lines: {len(cleaned_lines)}")
            return True
            
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return False
    
    def analyze_dataset(self):
        """Comprehensive analysis of the dataset"""
        print("\n" + "=" * 60)
        print("DATASET ANALYSIS")
        print("=" * 60)
        
        try:
            with open(self.cleaned_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic statistics
            lines = content.split('\n')
            sentences = sent_tokenize(content)
            words = word_tokenize(content.lower())
            
            # Remove punctuation for word analysis
            words_clean = [word for word in words if word.isalpha()]
            
            # Calculate statistics
            stats = {
                'total_characters': len(content),
                'total_lines': len(lines),
                'total_sentences': len(sentences),
                'total_words': len(words),
                'total_clean_words': len(words_clean),
                'unique_words': len(set(words_clean)),
                'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
                'avg_line_length': len(words) / len(lines) if lines else 0,
                'vocabulary_diversity': len(set(words_clean)) / len(words_clean) if words_clean else 0
            }
            
            # Print basic statistics
            print("BASIC STATISTICS:")
            print(f"   Total characters: {stats['total_characters']:,}")
            print(f"   Total lines: {stats['total_lines']:,}")
            print(f"   Total sentences: {stats['total_sentences']:,}")
            print(f"   Total words: {stats['total_words']:,}")
            print(f"   Clean words (alphabetic): {stats['total_clean_words']:,}")
            print(f"   Unique words: {stats['unique_words']:,}")
            print(f"   Average sentence length: {stats['avg_sentence_length']:.1f} words")
            print(f"   Average line length: {stats['avg_line_length']:.1f} words")
            print(f"   Vocabulary diversity: {stats['vocabulary_diversity']:.4f}")
            
            # Word frequency analysis
            word_freq = Counter(words_clean)
            most_common = word_freq.most_common(20)
            
            print("\nMOST COMMON WORDS:")
            for i, (word, count) in enumerate(most_common, 1):
                percentage = (count / len(words_clean)) * 100
                print(f"   {i:2d}. '{word}': {count:,} times ({percentage:.2f}%)")
            
            # Character analysis
            char_freq = Counter(content.lower())
            char_stats = {char: count for char, count in char_freq.items() if char.isalpha()}
            total_chars = sum(char_stats.values())
            
            print(f"\nCHARACTER FREQUENCY (Top 10):")
            for char, count in sorted(char_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                percentage = (count / total_chars) * 100
                print(f"   '{char}': {count:,} times ({percentage:.2f}%)")
            
            # Sentence length distribution
            sentence_lengths = [len(word_tokenize(sent)) for sent in sentences]
            
            print(f"\nSENTENCE LENGTH STATISTICS:")
            print(f"   Shortest sentence: {min(sentence_lengths)} words")
            print(f"   Longest sentence: {max(sentence_lengths)} words")
            print(f"   Median sentence length: {np.median(sentence_lengths):.1f} words")
            print(f"   Standard deviation: {np.std(sentence_lengths):.1f} words")
            
            # Save statistics to file
            self.save_statistics(stats, word_freq, char_stats, sentence_lengths)
            
            # Create visualizations
            self.create_visualizations(word_freq, char_stats, sentence_lengths)
            
            return stats
            
        except Exception as e:
            print(f"Error analyzing dataset: {e}")
            return None
    
    def save_statistics(self, stats, word_freq, char_stats, sentence_lengths):
        """Save detailed statistics to files"""
        print("\nSAVING STATISTICS...")
        
        # Save basic statistics
        with open('dataset_statistics.txt', 'w') as f:
            f.write("SHERLOCK HOLMES DATASET STATISTICS\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\n\nMOST COMMON WORDS:\n")
            f.write("-" * 20 + "\n")
            for word, count in word_freq.most_common(50):
                f.write(f"{word}: {count}\n")
        
        # Save word frequency as CSV
        word_df = pd.DataFrame(word_freq.most_common(100), columns=['word', 'frequency'])
        word_df.to_csv('word_frequency.csv', index=False)
        
        # Save sentence length statistics
        sent_df = pd.DataFrame(sentence_lengths, columns=['length'])
        sent_df.to_csv('sentence_lengths.csv', index=False)
        
        print("Statistics saved to current directory")
    
    def create_visualizations(self, word_freq, char_stats, sentence_lengths):
        """Create visualizations of the dataset"""
        print("\nCREATING VISUALIZATIONS...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Top 20 word frequencies
        top_words = dict(word_freq.most_common(20))
        axes[0, 0].barh(list(top_words.keys()), list(top_words.values()))
        axes[0, 0].set_title('Top 20 Most Frequent Words')
        axes[0, 0].set_xlabel('Frequency')
        
        # 2. Character frequency
        top_chars = dict(sorted(char_stats.items(), key=lambda x: x[1], reverse=True)[:10])
        axes[0, 1].bar(list(top_chars.keys()), list(top_chars.values()))
        axes[0, 1].set_title('Top 10 Character Frequencies')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Sentence length distribution
        axes[1, 0].hist(sentence_lengths, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Sentence Length Distribution')
        axes[1, 0].set_xlabel('Sentence Length (words)')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        axes[1, 1].imshow(wordcloud, interpolation='bilinear')
        axes[1, 1].set_title('Word Cloud')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualizations saved as dataset_analysis.png")
    
    def run_complete_analysis(self):
        """Run the complete data download and analysis pipeline"""
        print("STARTING COMPLETE DATA ANALYSIS PIPELINE")
        
        # Download data
        if not self.download_data():
            return False
        
        # Clean data
        if not self.clean_data():
            return False
        
        # Analyze dataset
        stats = self.analyze_dataset()
        
        if stats:
            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETE")
            print("=" * 60)
            print("Files created in current directory:")
            print("   dataset_statistics.txt - Detailed statistics")
            print("   word_frequency.csv - Word frequency data")
            print("   sentence_lengths.csv - Sentence length data")
            print("   dataset_analysis.png - Visualizations")
            print("   sherlock_holmes_cleaned.txt - Cleaned dataset")
            return True
        
        return False

if __name__ == "__main__":
    analyzer = SherlockHolmesDataAnalyzer()
    analyzer.run_complete_analysis()
