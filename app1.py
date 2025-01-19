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
    return render_template("face.html")

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
