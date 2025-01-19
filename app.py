<<<<<<< HEAD
import logging
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import lyricsgenius
import pandas as pd
import pickle
import re
from flask import Flask, request, render_template

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Replace with your Genius API Access Token
GENIUS_ACCESS_TOKEN = "Qb95YJ2NpRCxYMB7At9ulTEnxr_HLflxACNFAIkgD9e1JdV4EjbaYzJxRO7heDFn"

# Initialize the Genius API client
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)

# Load models
df = pickle.load(open('df.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Spotify API credentials
client_id = "a339cc448e49480e940ce1ebd8ddd9b3"
client_secret = "4ea98e0f9a0647998d82c8edf89bf59a"

# Authenticate with Spotify API
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Function to search songs on Spotify
def search_songs(query, limit=10):
    results = sp.search(q=query, type='track', limit=limit)
    tracks = []

    seen_tracks = set()  # To track unique tracks

    for item in results['tracks']['items']:
        track = {
            'id': item['id'],
            'name': item['name'],
            'artist': ", ".join([artist['name'] for artist in item['artists']]),
            'album': item['album']['name'],
            'release_date': item['album']['release_date'],
            'popularity': item['popularity']
            
        }

        # Clean song name and artist for deduplication
        cleaned_name = re.sub(r'(\(.*\))', '', track['name']).strip()
        cleaned_artist = re.sub(r'(\[.*\])', '', track['artist']).strip()

        # Create a unique signature
        track_signature = (cleaned_name, cleaned_artist, track['album'], track['release_date'])

        if track_signature not in seen_tracks:
            tracks.append(track)
            seen_tracks.add(track_signature)

    return tracks

def get_song_metadata(track_id):
    features = sp.audio_features(track_id)[0]  # Fetch audio features for the track
    metadata = {
        'danceability': features['danceability'],
        'energy': features['energy'],
        'tempo': features['tempo'],
        'key': features['key'],
        'mode': features['mode'],  # Mode 0 = Minor, 1 = Major
        'acousticness': features['acousticness'],
        'instrumentalness': features['instrumentalness'],
        'liveness': features['liveness'],
        'valence': features['valence'],
    }
    return metadata


# Function to fetch lyrics using Genius API
def fetch_lyrics(spotify_track_name):
    genius_songs = genius.search(spotify_track_name, per_page=1)
    logging.debug("Genius Search Results: %s", genius_songs)
    if genius_songs and len(genius_songs) > 0:
        song_id = genius_songs[0]['id']
        lyrics = genius.song(song_id)['song']['lyrics'] if 'lyrics' in genius.song(song_id)['song'] else "Lyrics not found"
    else:
        lyrics = "Lyrics not found"
    return lyrics

# Function to recommend songs
def recommendation(user_song, spotify_limit=10):
    # Search songs from Spotify
    spotify_songs = search_songs(user_song, limit=spotify_limit)

    if not spotify_songs:
        return None  # No results from Spotify
    
    recommendations = []
    for song in spotify_songs:
        song_name = song['name'].strip()  # Clean song name
        artist_name = ", ".join(artist.strip() for artist in song['artist'].split(','))  # Clean artist name(s)
        song_link = f"https://open.spotify.com/track/{song['id']}"  # Spotify link
        
        # Format output as [Song Name] by [Artist Name]
        formatted_output = f"<a href='{song_link}' target='_blank'>{song_name} by {artist_name}</a>"
        
        recommendations.append(formatted_output)

    return recommendations




# Function to fetch combined data (Spotify and Genius)
def fetch_combined_data(song_query, spotify_limit=10, genius_limit=1):
    # Fetch data from Spotify
    spotify_songs = search_songs(song_query, limit=spotify_limit)
    combined_data = []

    for song in spotify_songs:
        # Fetch Genius lyrics if available
        genius_songs = genius.search(song['name'], per_page=genius_limit)
        if genius_songs:
            genius_song = genius_songs[0]  # Use the first Genius result
            lyrics = fetch_lyrics(genius_song['name'])  # Fetch lyrics from Genius
        else:
            lyrics = "Lyrics not found"

        # Add Spotify and Genius data to the combined dataset
        song_data = {
            'spotify_id': song['id'],
            'name': song['name'],
            'artist': song['artist'],
            'album': song['album'],
            'release_date': song['release_date'],
            'popularity': song['popularity'],
            'lyrics': lyrics
        }
        combined_data.append(song_data)

    return pd.DataFrame(combined_data)

# Initialize Flask app
app = Flask(__name__)

# Route to render the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle song recommendation request
@app.route('/recom', methods=['POST'])
def recommend_songs():
    user_song = request.form.get('song_name')  # Safely get the song name from form input

    if not user_song:
        return render_template('index.html', error="Please provide a song name.")
    
    user_song = user_song.strip()
    songs = recommendation(user_song)

    if songs:
        return render_template('index.html', selected_song=user_song, songs=songs)
    else:
        return render_template('index.html', error="Song not found! Please try a different name.")

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=8000)
=======
from flask import Flask, render_template, request, jsonify
import cv2
import os
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import base64
import random

app = Flask(__name__)

# Spotify API Credentials
SPOTIFY_CLIENT_ID = "0869bee7da374190902e03212551dc13"
SPOTIFY_CLIENT_SECRET = "4812f8afcd444f698cc612c82bb3fcbd"
REDIRECT_URI = 'http://localhost:5000'
SPOTIFY_API_URL = 'https://api.spotify.com/v1'

# Face++ API Credentials
FACE_API_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"
FACE_API_KEY = "mNN7hHxvFE8WJL-P6WU3IeenMVgFwdT4"
FACE_API_SECRET = "UNIoreteGMK8guoI0f56EGNfDlOFH_4o"

# Route: Home
@app.route("/")
def index():
    return render_template("index.html")

# Route: Capture Image
@app.route("/capture", methods=["POST"])
def capture_image():
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Failed to open the camera."}), 500

    ret, frame = cap.read()
    if ret:
        cv2.imwrite("face_image.jpg", frame)
        cap.release()
        cv2.destroyAllWindows()
        return jsonify({"message": "Image captured successfully.", "image_path": "face_image.jpg"})
    else:
        cap.release()
        cv2.destroyAllWindows()
        return jsonify({"error": "Failed to capture the image."}), 500

# Route: Preprocess Image
@app.route("/preprocess", methods=["POST"])
def preprocess_image():
    image_path = "face_image.jpg"
    if not os.path.exists(image_path):
        return jsonify({"error": "Captured image not found."}), 404

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return jsonify({"error": "No faces detected during preprocessing."}), 400

    # Crop the first detected face
    x, y, w, h = faces[0]
    face = image[y:y+h, x:x+w]
    cv2.imwrite("preprocessed_face.jpg", face)
    return jsonify({"message": "Image preprocessed successfully.", "image_path": "preprocessed_face.jpg"})

# Route: Detect Emotion
@app.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    image_path = "preprocessed_face.jpg"
    if not os.path.exists(image_path):
        return jsonify({"error": "Preprocessed image not found."}), 404

    with open(image_path, 'rb') as image_file:
        files = {'image_file': image_file}
        data = {
            'api_key': FACE_API_KEY,
            'api_secret': FACE_API_SECRET,
            'return_attributes': 'emotion'
        }
        response = requests.post(FACE_API_URL, data=data, files=files)
        result = response.json()

        if response.status_code == 200:
            if "faces" in result and len(result["faces"]) > 0:
                emotions = result['faces'][0]['attributes']['emotion']
                dominant_emotion = max(emotions, key=emotions.get)
                return jsonify({"emotions": emotions, "dominant_emotion": dominant_emotion})
            else:
                return jsonify({"error": "No faces detected."}), 400
        else:
            return jsonify({"error": result.get('error_message', 'Unknown error')}), 500

# Get Spotify Access Token using Client Credentials Flow
def get_access_token():
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_data = {
        'grant_type': 'client_credentials',
    }
    auth_headers = {
        'Authorization': 'Basic ' + base64.b64encode(f'{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}'.encode()).decode()
    }
    
    response = requests.post(auth_url, data=auth_data, headers=auth_headers)
    return response.json().get('access_token')

# Route: Recommend Music
@app.route("/recommend_music", methods=["POST"])
def recommend_music():
    emotion = request.json.get('dominant_emotion')
    if emotion:
        access_token = get_access_token()
        recommended_songs = fetch_music_based_on_emotion(emotion, access_token)
        return jsonify({'tracks': recommended_songs})
    else:
        return jsonify({'error': 'No emotion detected'}), 400

# Function to search Spotify tracks based on detected emotion
def fetch_music_based_on_emotion(emotion, access_token):
    query = ''
    
    if emotion == 'happy':
        query = 'genre:party OR genre:happy OR genre:upbeat'
    elif emotion == 'sadness':
        query = 'genre:calm OR genre:chill OR genre:sad OR genre:melancholy'
    elif emotion == 'neutral':
        query = 'genre:lofi OR genre:chill OR genre:instrumental OR genre:acoustic'
    elif emotion == 'anger':
        query = 'genre:rap OR genre:metal OR genre:hard rock OR genre:punk'
    elif emotion == 'disgust':
        query = 'genre:dance pop OR genre:funk OR genre:happy'
    elif emotion == 'fear':
        query = 'genre:calm OR genre:jazz OR genre:meditation OR genre:empowering'
    elif emotion == 'surprise':
        query = 'genre:disco OR genre:electronic OR genre:funk OR genre:experimental'
    else:
        query = 'genre:pop'

    search_url = f'{SPOTIFY_API_URL}/search'
    params = {
        'q': query,
        'type': 'track',
        'limit': 10  # Fetch 5 tracks as an example
    }
    
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    
    response = requests.get(search_url, params=params, headers=headers)
    tracks = response.json().get('tracks', {}).get('items', [])
    music_data = []
    
    for track in tracks:
        music_data.append({
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'url': track['external_urls']['spotify'],  # Direct URL to play the track
            'image': track['album']['images'][0]['url']  # Track image
        })
    
    return music_data

# Run the Flask server
if __name__ == "__main__":
    app.run(debug=True)
>>>>>>> 959822c7f7afc54d2162c8e4cee71b9f2b8c5019
