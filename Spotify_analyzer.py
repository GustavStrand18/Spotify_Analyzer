"""
Spotify Kaggle Data Analyzer - Medium Version
============================================
Analyserer Spotify data fra Kaggle dataset og viser interessante mønstre.

Dataset: "30000 Spotify Songs" eller "Most Streamed Spotify Songs 2023"
Krav: pip install matplotlib pandas seaborn
Data: Download fra Kaggle og gem som CSV fil
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class SpotifyDataLoader:
    """Håndterer indlæsning af Spotify CSV data fra Kaggle"""
    
    def __init__(self, csv_file=None):
        # Automatisk find CSV filer i mappen
        if csv_file is None:
            csv_file = self._find_spotify_csv()
        self.csv_file = Path(csv_file) if csv_file else None
        self.data = None
    
    def _find_spotify_csv(self):
        """Find Spotify CSV fil automatisk"""
        possible_files = [
            "spotify_30k_songs.csv",      # 30000 Spotify Songs
            "spotify_top_2023.csv",       # Top Spotify Songs 2023  
            "spotify_data.csv",           # Generisk navn
            "30000-spotify-songs.csv",    # Original navn fra Kaggle
            "top-spotify-songs-2023.csv"  # Original navn fra Kaggle
        ]
        
        for filename in possible_files:
            if Path(filename).exists():
                print(f"🎵 Fandt automatisk: {filename}")
                return filename
        
        return "spotify_data.csv"  # Default fallback
    
    def load_data(self):
        """Indlæs CSV data med error handling"""
        try:
            if not self.csv_file.exists():
                print(f"❌ Filen {self.csv_file} blev ikke fundet!")
                print("\n📥 Download data fra:")
                print("1. https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs")
                print("2. https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023")
                print("3. Gem som 'spotify_data.csv' i samme mappe som scriptet")
                print("\n🧪 Eller vil du teste med simulerede data? (j/n): ", end="")
                
                # Simpel test data hvis filen ikke findes
                choice = input().lower()
                if choice == 'j' or choice == 'y':
                    return self._create_sample_data()
                else:
                    return None
            
            # Prøv forskellige encoding for at håndtere special karakterer
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    self.data = pd.read_csv(self.csv_file, encoding=encoding)
                    print(f"✅ Indlæste {len(self.data)} sange med {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.data is None:
                raise Exception("Kunne ikke læse CSV fil med nogen encoding")
            
            # Print column info for debugging
            print(f"📊 Kolonner i datasæt: {list(self.data.columns[:10])}...")
            return self.data
            
        except Exception as e:
            print(f"❌ Fejl ved indlæsning: {e}")
            return None
    
    def _create_sample_data(self):
        """Lav test data til demonstration"""
        print("🧪 Genererer test data...")
        
        artists = ["Taylor Swift", "Drake", "Ed Sheeran", "Billie Eilish", "The Weeknd", 
                  "Ariana Grande", "Post Malone", "Dua Lipa", "Harry Styles", "Olivia Rodrigo"]
        genres = ["pop", "hip-hop", "rock", "electronic", "indie", "r&b"]
        
        import random
        
        sample_data = []
        for i in range(1000):  # 1000 test sange
            song_data = {
                'track_name': f"Song {i+1}",
                'track_artist': random.choice(artists),
                'track_genre': random.choice(genres),
                'track_popularity': random.randint(20, 100),
                'danceability': round(random.uniform(0.1, 1.0), 3),
                'energy': round(random.uniform(0.1, 1.0), 3),
                'valence': round(random.uniform(0.1, 1.0), 3),
                'tempo': random.randint(60, 180),
                'acousticness': round(random.uniform(0.0, 1.0), 3),
                'speechiness': round(random.uniform(0.0, 0.5), 3)
            }
            sample_data.append(song_data)
        
        self.data = pd.DataFrame(sample_data)
        print(f"✅ Genererede {len(self.data)} test sange")
        return self.data
    
    def get_data_info(self):
        """Få information om datasættet"""
        if self.data is None:
            return None
        
        info = {
            'total_songs': len(self.data),
            'columns': list(self.data.columns),
            'shape': self.data.shape,
            'memory_usage': f"{self.data.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
        }
        return info


class SpotifyAnalyzer:
    """Hovedklasse til analyse af Kaggle Spotify data"""
    
    def __init__(self, data):
        self.data = data
        self.processed_data = self._preprocess_data()
    
    def _preprocess_data(self):
        """Forbehandle data og håndter forskellige kolonnenavne"""
        if self.data is None or self.data.empty:
            return pd.DataFrame()
        
        df = self.data.copy()
        
        # Standardiser kolonnenavne (forskellige datasets har forskellige navne)
        column_mapping = {
            'track_name': 'song',
            'track.name': 'song', 
            'name': 'song',
            'artist_name': 'artist',
            'artists': 'artist',
            'track_artist': 'artist',
            'artist.name': 'artist',
            'track_genre': 'genre',
            'playlist_genre': 'genre',
            'genres': 'genre',
            'track_popularity': 'popularity',
            'popularity': 'popularity',
            'streams': 'stream_count'
        }
        
        # Rename kolonner hvis de findes
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Fjern missing values for vigtige kolonner
        essential_cols = ['song', 'artist']
        for col in essential_cols:
            if col in df.columns:
                df = df.dropna(subset=[col])
        
        return df
    
    def get_top_artists(self, n=10):
        """Find top N artister baseret på antal sange eller streams"""
        if self.processed_data.empty:
            return pd.DataFrame()
        
        if 'artist' not in self.processed_data.columns:
            return pd.DataFrame()
        
        # Tæl antal sange per artist
        artist_counts = self.processed_data['artist'].value_counts().head(n)
        
        # Hvis vi har popularity eller streams data, brug det også
        result_data = {'song_count': artist_counts}
        
        if 'popularity' in self.processed_data.columns:
            avg_popularity = self.processed_data.groupby('artist')['popularity'].mean()
            result_data['avg_popularity'] = avg_popularity
        
        if 'stream_count' in self.processed_data.columns:
            total_streams = self.processed_data.groupby('artist')['stream_count'].sum()
            result_data['total_streams'] = total_streams
        
        result_df = pd.DataFrame(result_data)
        return result_df.sort_values(result_df.columns[0], ascending=False).head(n)
    
    def get_top_songs(self, n=10):
        """Find top N sange baseret på popularitet eller streams"""
        if self.processed_data.empty:
            return pd.DataFrame()
        
        # Prioriter streams, så popularity, så sang count
        if 'stream_count' in self.processed_data.columns:
            return self.processed_data.nlargest(n, 'stream_count')[['song', 'artist', 'stream_count']]
        elif 'popularity' in self.processed_data.columns:
            return self.processed_data.nlargest(n, 'popularity')[['song', 'artist', 'popularity']]
        else:
            # Bare vis de første sange
            return self.processed_data[['song', 'artist']].head(n)
    
    def analyze_audio_features(self):
        """Analyser audio features som energy, danceability, etc."""
        audio_features = ['danceability', 'energy', 'speechiness', 'acousticness', 
                         'instrumentalness', 'liveness', 'valence', 'tempo']
        
        available_features = [f for f in audio_features if f in self.processed_data.columns]
        
        if not available_features:
            return {}
        
        stats = {}
        for feature in available_features:
            stats[feature] = {
                'mean': self.processed_data[feature].mean(),
                'std': self.processed_data[feature].std(),
                'min': self.processed_data[feature].min(),
                'max': self.processed_data[feature].max()
            }
        
        return stats
    
    def analyze_genres(self, n=10):
        """Analyser genre distribution"""
        if 'genre' not in self.processed_data.columns:
            return pd.Series()
        
        # Håndter tilfælde hvor genre er en komma-separeret string
        all_genres = []
        for genre_entry in self.processed_data['genre'].dropna():
            if isinstance(genre_entry, str) and ',' in genre_entry:
                genres = [g.strip() for g in genre_entry.split(',')]
                all_genres.extend(genres)
            else:
                all_genres.append(str(genre_entry))
        
        return pd.Series(all_genres).value_counts().head(n)
    
    def get_dataset_stats(self):
        """Få overordnede statistikker om datasættet"""
        if self.processed_data.empty:
            return {}
        
        stats = {
            'total_songs': len(self.processed_data),
            'unique_artists': self.processed_data['artist'].nunique() if 'artist' in self.processed_data.columns else 0,
        }
        
        # Tilføj stats baseret på tilgængelige kolonner
        if 'popularity' in self.processed_data.columns:
            stats['avg_popularity'] = round(self.processed_data['popularity'].mean(), 2)
        
        if 'stream_count' in self.processed_data.columns:
            stats['total_streams'] = self.processed_data['stream_count'].sum()
        
        if 'genre' in self.processed_data.columns:
            stats['unique_genres'] = self.processed_data['genre'].nunique()
        
        return stats


class SpotifyVisualizer:
    """Håndterer visualisering af Kaggle Spotify data"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
        # Sæt Spotify-inspireret stil
        plt.style.use('dark_background')
        self.spotify_green = '#1DB954'
        self.spotify_black = "#130F0F"
    
    def plot_top_artists(self, n=10):
        """Plot top artister"""
        top_artists = self.analyzer.get_top_artists(n)
        
        if top_artists.empty:
            print("Ingen artist data at plotte")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Brug første kolonne (normalt song_count)
        first_col = top_artists.columns[0]
        bars = plt.barh(range(len(top_artists)), top_artists[first_col], color=self.spotify_green)
        
        plt.yticks(range(len(top_artists)), top_artists.index)
        plt.xlabel(f'{first_col.replace("_", " ").title()}')
        plt.title(f'Top {n} Artister', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Tilføj værdier på bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def plot_audio_features(self):
        """Plot audio features radar chart"""
        audio_stats = self.analyzer.analyze_audio_features()
        
        if not audio_stats:
            print("Ingen audio features tilgængelige")
            return
        
        # Lav et heatmap i stedet for radar chart (nemmere)
        features = list(audio_stats.keys())[:8]  # Max 8 features
        means = [audio_stats[f]['mean'] for f in features]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(features, means, color=self.spotify_green, alpha=0.7)
        plt.title('Gennemsnitlige Audio Features', fontsize=16, fontweight='bold')
        plt.ylabel('Værdi')
        plt.xticks(rotation=45)
        
        # Tilføj værdier på bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_genre_distribution(self, n=10):
        """Plot genre fordeling"""
        genre_counts = self.analyzer.analyze_genres(n)
        
        if genre_counts.empty:
            print("Ingen genre data tilgængelig")
            return
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(genre_counts)), genre_counts.values, color=self.spotify_green)
        plt.xticks(range(len(genre_counts)), genre_counts.index, rotation=45, ha='right')
        plt.ylabel('Antal sange')
        plt.title(f'Top {n} Genrer', fontsize=16, fontweight='bold')
        
        # Tilføj værdier
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def create_dashboard(self):
        """Lav et samlet dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🎵 Spotify Data Dashboard 🎵', fontsize=20, fontweight='bold')
        
        # 1. Top artister
        top_artists = self.analyzer.get_top_artists(8)
        if not top_artists.empty:
            first_col = top_artists.columns[0]
            axes[0,0].barh(range(len(top_artists)), top_artists[first_col], color=self.spotify_green)
            axes[0,0].set_yticks(range(len(top_artists)))
            axes[0,0].set_yticklabels(top_artists.index)
            axes[0,0].set_title('Top Artister')
            axes[0,0].invert_yaxis()
        
        # 2. Genre fordeling
        genres = self.analyzer.analyze_genres(6)
        if not genres.empty:
            axes[0,1].pie(genres.values, labels=genres.index, autopct='%1.1f%%', 
                         colors=plt.cm.Greens(np.linspace(0.4, 0.8, len(genres))))
            axes[0,1].set_title('Genre Fordeling')
        
        # 3. Audio features
        audio_stats = self.analyzer.analyze_audio_features()
        if audio_stats:
            features = list(audio_stats.keys())[:6]
            means = [audio_stats[f]['mean'] for f in features]
            axes[1,0].bar(features, means, color=self.spotify_green, alpha=0.7)
            axes[1,0].set_title('Audio Features')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Dataset statistikker
        stats = self.analyzer.get_dataset_stats()
        if stats:
            axes[1,1].axis('off')
            stats_text = "📊 DATASET STATISTIKKER\n\n"
            for key, value in stats.items():
                formatted_key = key.replace('_', ' ').title()
                if isinstance(value, (int, float)) and value > 1000:
                    formatted_value = f"{value:,}"
                else:
                    formatted_value = str(value)
                stats_text += f"{formatted_key}: {formatted_value}\n"
            
            axes[1,1].text(0.1, 0.5, stats_text, fontsize=12, 
                          verticalalignment='center', color='white',
                          bbox=dict(boxstyle="round,pad=0.5", facecolor=self.spotify_black))
        
        plt.tight_layout()
        plt.show()


def main():
    """Hovedprogram"""
    print("🎵 SPOTIFY KAGGLE DATA ANALYZER 🎵")
    print("=" * 45)
    
    # Indlæs data
    print("\n📂 Indlæser Kaggle data...")
    loader = SpotifyDataLoader("spotify_data.csv")  
    data = loader.load_data()
    
    if data is None:
        print("\n❌ Kunne ikke indlæse data!")
        print("\n📥 Sådan får du data:")
        print("1. Gå til Kaggle.com og søg 'spotify songs'")
        print("2. Download '30000 Spotify Songs' eller 'Top Spotify Songs 2023'")
        print("3. Gem CSV filen som 'spotify_data.csv' i samme mappe som dette script")
        return
    
    # Vis data info
    info = loader.get_data_info()
    if info:
        print(f"\n📊 Dataset info:")
        print(f"🎵 Total sange: {info['total_songs']:,}")
        print(f"📏 Dimensioner: {info['shape']}")
        print(f"💾 Hukommelsesforbrug: {info['memory_usage']}")
    
    # Analyser data
    print(f"\n🔍 Analyserer data...")
    analyzer = SpotifyAnalyzer(data)
    
    # Print statistikker
    stats = analyzer.get_dataset_stats()
    print(f"\n📈 DATASET STATISTIKKER:")
    for key, value in stats.items():
        formatted_key = key.replace('_', ' ').title()
        print(f"🎵 {formatted_key}: {value:,}" if isinstance(value, (int, float)) else f"🎵 {formatted_key}: {value}")
    
        
    # Top artister
    print(f"\n🌟 TOP 5 ARTISTER:")
    top_artists = analyzer.get_top_artists(5)
    if not top_artists.empty:
        first_col = top_artists.columns[0]
        for i, (artist, data) in enumerate(top_artists.iterrows(), 1):
            print(f"{i}. {artist}: {data[first_col]} {first_col.replace('_', ' ')}")
    
    # Visualiseringer
    print(f"\n📊 Genererer visualiseringer...")
    visualizer = SpotifyVisualizer(analyzer)
    
    # Vis plots
    visualizer.plot_top_artists(10)
    visualizer.plot_audio_features()
    visualizer.plot_genre_distribution(8)
    visualizer.create_dashboard()
    
    print(f"\n✅ Analyse færdig! Tjek dine grafer 📈")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Tak for at bruge Spotify Analyzer!")
    except Exception as e:
        print(f"\n❌ Der skete en fejl: {e}")
        print("Tjek at du har installeret: pip install matplotlib pandas seaborn")