"""
generate_synthetic_data.py: Generates a realistic synthetic Spotify Playlists dataset.

Matches the exact Kaggle Larxel dataset schema:
Columns: user_id, "artistname", "trackname", "playlistname"
Features:
- Quoted strings with backslash escapes
- Redundant track edition suffixes (- Remastered 2011, - Live, - Radio Edit)
- Realistic musical clusters (Britpop, Classic Rock, 90s Grunge, 2010s Pop, EDM)
- Multiple playlists per user, including duplicate named playlists ("Favorites", "Starred")
- Skewed playlist lengths (long tail distribution)
"""

import csv
import random
from pathlib import Path

# Music Clusters to simulate realistic association rules
CLUSTERS = {
    "britpop": [
        ("Oasis", "Wonderwall - Remastered"),
        ("Oasis", "Don't Look Back In Anger - Remastered 2014"),
        ("Oasis", "Champagne Supernova"),
        ("The Verve", "Bitter Sweet Symphony - Radio Edit"),
        ("The Verve", "Lucky Man"),
        ("Blur", "Song 2"),
        ("Blur", "Parklife"),
        ("Pulp", "Common People - Single Version"),
    ],
    "classic_rock": [
        ("Queen", "Bohemian Rhapsody - 2011 Mix"),
        ("Queen", "Don't Stop Me Now - Remastered"),
        ("Queen", "Another One Bites The Dust"),
        ("Led Zeppelin", "Stairway to Heaven - Remaster"),
        ("Pink Floyd", "Comfortably Numb - Live"),
        ("The Eagles", "Hotel California - Remastered 2013"),
        ("Fleetwood Mac", "Dreams - 2004 Remaster"),
        ("Fleetwood Mac", "Go Your Own Way"),
    ],
    "grunge": [
        ("Nirvana", "Smells Like Teen Spirit"),
        ("Nirvana", "Come As You Are - Remastered"),
        ("Pearl Jam", "Alive - Live"),
        ("Pearl Jam", "Jeremy - Remastered 2011"),
        ("Soundgarden", "Black Hole Sun - Radio Edit"),
        ("Alice in Chains", "Rooster"),
        ("Foo Fighters", "Everlong"),
    ],
    "pop_2010s": [
        ("Taylor Swift", "Blank Space"),
        ("Taylor Swift", "Shake It Off"),
        ("Katy Perry", "Roar"),
        ("Katy Perry", "Firework"),
        ("Ed Sheeran", "Shape of You"),
        ("Ed Sheeran", "Thinking Out Loud"),
        ("Dua Lipa", "Levitating - Single Version"),
        ("The Weeknd", "Blinding Lights"),
    ],
    "edm": [
        ("Avicii", "Wake Me Up - Radio Edit"),
        ("Avicii", "Levels - Original Mix"),
        ("Calvin Harris", "Summer"),
        ("Calvin Harris", "Feel So Close - Radio Edit"),
        ("David Guetta", "Titanium"),
        ("Swedish House Mafia", "Don't You Worry Child"),
        ("Tiësto", "The Business"),
    ]
}

ALL_TRACKS = [trk for cluster in CLUSTERS.values() for trk in cluster]


def generate_synthetic_csv(output_path: Path, num_users=150, playlists_per_user=(2, 6)):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    rows = []
    cluster_names = list(CLUSTERS.keys())

    for u_idx in range(1, num_users + 1):
        user_id = f"user_{u_idx:04d}"
        n_playlists = random.randint(*playlists_per_user)

        # Primary user preference
        user_affinity = random.sample(cluster_names, k=min(2, len(cluster_names)))

        for p_idx in range(1, n_playlists + 1):
            # Mix generic names ("Starred", "Chill", "Favorites") and themed names
            generic_names = ["Starred", "Favorites", "My Playlist", "Chill", "Driving"]
            if random.random() < 0.4:
                playlist_name = random.choice(generic_names)
            else:
                playlist_name = f"Vibes {random.choice(user_affinity)}_{p_idx}"

            # Playlist size: skewed lognormal
            p_len = max(3, min(60, int(random.lognormvariate(2.2, 0.6))))

            # Pick items heavily weighted towards affinity cluster
            playlist_tracks = set()
            for _ in range(p_len * 2):
                if len(playlist_tracks) >= p_len:
                    break
                if random.random() < 0.7:
                    # Pick from affinity
                    aff = random.choice(user_affinity)
                    playlist_tracks.add(random.choice(CLUSTERS[aff]))
                else:
                    # Pick random
                    playlist_tracks.add(random.choice(ALL_TRACKS))

            for artist, track in playlist_tracks:
                rows.append([user_id, artist, track, playlist_name])

    # Shuffle rows to simulate real logging
    random.shuffle(rows)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
        # Exact headers
        writer.writerow(["user_id", "artistname", "trackname", "playlistname"])
        writer.writerows(rows)

    print(f"[synthetic] Generated {len(rows):,} rows across {num_users} users in: {output_path}")
    return output_path


if __name__ == "__main__":
    from config import RAW_DATA_DIR
    target_csv = RAW_DATA_DIR / "synthetic_spotify_playlists.csv"
    generate_synthetic_csv(target_csv)
