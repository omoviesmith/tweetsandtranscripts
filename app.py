from flask import Flask, request, jsonify, Response, session, make_response
from flask.sessions import SecureCookieSessionInterface
from flask_cors import CORS, cross_origin
import youtube_dl
import yt_dlp as youtube_dl
import assemblyai as aai
import os
import datetime
from flask import Flask, jsonify, request
from snscrape import main
from dotenv import load_dotenv
import boto3
import csv
import io
# import good
import logging
import sys
# from upload_utils import upload_to_s3

# Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# AWS related variables
s3_bucket = 'twitsbucket'
folder = 'transcripts'

load_dotenv()  # take environment variables from .env.
bearer_token = os.getenv("BEARER_TOKEN")
AWS_ACCESS_KEY_ID = os.getenv("ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("REGION")
# Assuming the AssemblyAI API key is set as an environment variable
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

# creating AWS clients
s3 = boto3.client('s3', 
                        aws_access_key_id=AWS_ACCESS_KEY_ID, 
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                        region_name=AWS_REGION)

# method to upload the files to s3 bucket
def upload_to_s3(file_content, object_name):
    s3_client = boto3.client('s3', 
                             aws_access_key_id=AWS_ACCESS_KEY_ID, 
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                             region_name=AWS_REGION)
    
    default_bucket_name = "twittrans"
    key = f'transcriptions/{object_name}'
    
    try:
        # Assuming file_content is an io.StringIO object
        # Convert StringIO content to bytes using getvalue and encode
        file_content_bytes = file_content.getvalue().encode()  # Default encode to utf-8
        file_content_bytes_io = io.BytesIO(file_content_bytes)  # Convert bytes to a BytesIO object

        # Upload the BytesIO object, which is effectively the "file"
        s3_client.upload_fileobj(file_content_bytes_io, default_bucket_name, key)
        s3_url = f"https://{default_bucket_name}.s3.amazonaws.com/{key}"


        return s3_url

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
# method to extract audio from youtube video
def parse_s3_url(s3_url):
    # Assuming the S3 URL is in the standard format of 's3://bucket_name/object_key'
    # Extract the bucket name and object key from the S3 URL
    try:
        bucket_name, object_key = s3_url.replace("s3://", "").split('/', 1)
        return bucket_name, object_key
        print(f"Parsed: Bucket name {bucket_name} and Object key {object_key}")
    except ValueError:
        raise ValueError("Invalid S3 URL format")

def extract_audio_from_yt_video(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        # Changed the 'outtmpl' to store the file with '.wav' extension
        'outtmpl': '%(id)s.%(ext)s',  
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            # Changed 'preferredcodec' to 'wav' 
            'preferredcodec': 'mp3', 
            'preferredquality': '192',
        }],
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info_dict)
        # Changed to '.wav' to match the postprocessor codec
        filename_wav = f"{filename.rsplit('.', 1)[0]}.mp3"

        # Proceed to upload to S3
        default_bucket_name = "twittrans"
        key = f'audio/{filename_wav}'

        # Upload to S3
        with open(filename_wav, 'rb') as file:
            s3.upload_fileobj(file, default_bucket_name, key)
            s3_url = f"s3://{default_bucket_name}/{key}"
            

            bucket_name, object_key = parse_s3_url(s3_url)
            s3_http_url = f"https://{bucket_name}.s3.amazonaws.com/{object_key}"
            print(f"Uploaded: Bucket name {bucket_name} and Object key {object_key}")

    return s3_http_url

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')  # change this to a secure random string
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.secret_key = os.getenv('SECRET_KEY')  # set the SECRET_KEY environment variable before running your app
CORS(app)

@app.route('/')
@cross_origin(supports_credentials=True)  # Apply CORS to this specific route now
def hello_world():
    return {"Hello":"World"}



@app.route('/process_audio', methods=['POST'])
@cross_origin(supports_credentials=True)
def process():
    try:
        urls = request.json.get('url', [])
        diarization = request.json.get('diarization', False)
        
        # Ensure URLs are in a list
        if isinstance(urls, str):
            urls = [urls]
        elif not isinstance(urls, list):
            raise ValueError("URL should be a string or list of strings.")

        # Process each URL
        results = []
        for url in urls:
            print(f"Processing URL: {url}")
            
            # Extract audio from YouTube video and upload to S3
            s3_audio_url = extract_audio_from_yt_video(url)
            print(f"Audio file uploaded to S3: {s3_audio_url}")
            
            # Use AssemblyAI API to transcribe the audio
            config = aai.TranscriptionConfig(speaker_labels=diarization)
            transcript = aai.Transcriber().transcribe(s3_audio_url, config)
            
            # Concatenate transcriptions with speaker labels
            transcription = ''
            for utterance in transcript.utterances:
                transcription += f"Speaker {utterance.speaker}: {utterance.text}\n"
            
            # Add the transcription result for the current video URL to the results list
            results.append([url, transcription])

        # Generate CSV content:
        csv_content = io.StringIO()
        writer = csv.writer(csv_content)
        writer.writerow(["URL", "Transcription"])
        writer.writerows(results)
        csv_content.seek(0)

        # Generate and upload CSV to S3
        csv_filename = f"transcriptions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        s3_url = upload_to_s3(csv_content, csv_filename)
        
        return jsonify({'download_link': s3_url}), 200

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'An error occurred while processing the audio.'}), 500

# @app.route('/process_audio', methods=['POST'])
# @cross_origin(supports_credentials=True)
# def process():
#     try:
#         urls = request.json.get('url', [])
#         diarization = request.json.get('diarization', False)
        
#         # Ensure URLs are in a list
#         if isinstance(urls, str):
#             urls = [urls]
#         elif not isinstance(urls, list):
#             raise ValueError("URL should be a string or list of strings.")

#         # Process each URL
#         results = []
#         for url in urls:
#             print(f"Processing URL: {url}")
#             transcription = good.process_audio(url, diarization)
#             results.append([url, transcription])

#         # Generate CSV content:
#         csv_content = io.StringIO()
#         writer = csv.writer(csv_content)
#         writer.writerow(["URL", "Transcription"])
#         writer.writerows(results)
#         csv_content.seek(0)

#         # Generate and upload CSV to S3
#         csv_filename = f"transcriptions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
#         s3_url = upload_to_s3(csv_content, csv_filename)
        
#         return jsonify({'download_link': s3_url}), 200

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return jsonify({'error': 'An error occurred while processing the audio.'}), 500

# @app.route('/process_audio', methods=['POST'])
# @cross_origin(supports_credentials=True)  # Apply CORS to this specific route
# def process():
    
#     urls = request.json['url']
    
#     # If the URLs are not in a list, convert it to a list
#     diarization = request.json['diarization']
#     if isinstance(urls, str):
#         urls = [urls]

#     results = []

#     for url in urls:
#         print(url)
#         transcription = good.process_audio(url, diarization)
#         results.append([url, transcription])

#     # Generate CSV content:
#     csv_content = io.StringIO()
#     writer = csv.writer(csv_content)
#     writer.writerow(["URL", "Transcription"])
#     writer.writerows(results)
#     csv_content.seek(0)

#     # Generate file name
#     csv_filename = f"transcriptions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
#     # Upload CSV content to S3 and get the URL
#     s3_url = upload_to_s3(csv_content, csv_filename)
    
#     if s3_url:
#         return jsonify({'download_link': s3_url}), 200
#     else:
#         return jsonify({'error': 'Failed to upload file to S3.'}), 500
    
    


### Updated `/extract_tweets` Endpoint:

@app.route('/extract_tweets', methods=['GET', 'POST'])
@cross_origin(supports_credentials=True)  # Apply CORS to this specific route
def extract_tweets():
    
    try:
        data = request.json
        if 'username' in data:
            username = data['username']
        else:
            return jsonify({'error': 'Username not found in the request.'}), 400

        # Assuming the main() function can accept the writer object:
        csv_content = io.StringIO()

        # Execute main() function to write directly into csv_content StringIO
        main(username, writer=csv.writer(csv_content))

        csv_filename = f"{username}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_content.seek(0)

        # Upload CSV content to S3 and get the URL
        s3_url = upload_to_s3(csv_content, csv_filename)
        
        if s3_url:
            return jsonify({'download_link': s3_url}), 200
        else:
            return jsonify({'error': 'Failed to upload file to S3.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Configure Flask logger
    # handler = logging.StreamHandler(stream=sys.stdout)
    # handler.setLevel(logging.INFO)
    # app.logger.addHandler(handler)
    # app.logger.setLevel(logging.INFO)

    # app.run(debug=True, host='0.0.0.0')
    app.run(debug=False, threaded=False)