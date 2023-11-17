from flask import Flask, request, jsonify, make_response
import os
import datetime
from flask import Flask, jsonify, request
from snscrape import main
from dotenv import load_dotenv
import boto3
import csv
import io
import good
# from upload_utils import upload_to_s3

# AWS related variables
s3_bucket = 'twitsbucket'
folder = 'transcripts'

load_dotenv()  # take environment variables from .env.
bearer_token = os.getenv("BEARER_TOKEN")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

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
        s3_url = f"https://{default_bucket_name}.s3.{AWS_REGION}.amazonaws.com/{key}"

        return s3_url

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

app = Flask(__name__)

@app.route('/process_audio', methods=['POST'])
def process():
    urls = request.json['url']
    # If the URLs are not in a list, convert it to a list
    diarization = request.json['diarization']
    if isinstance(urls, str):
        urls = [urls]

    results = []

    for url in urls:
        transcription = good.process_audio(url, diarization)
        results.append([url, transcription])

    # Generate CSV content:
    csv_content = io.StringIO()
    writer = csv.writer(csv_content)
    writer.writerow(["URL", "Transcription"])
    writer.writerows(results)
    csv_content.seek(0)

    # Generate file name
    csv_filename = f"transcriptions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # Upload CSV content to S3 and get the URL
    s3_url = upload_to_s3(csv_content, csv_filename)
    if s3_url:
        return jsonify({'download_link': s3_url}), 200
    else:
        return jsonify({'error': 'Failed to upload file to S3.'}), 500


### Updated `/extract_tweets` Endpoint:

@app.route('/extract_tweets', methods=['GET', 'POST'])
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
    app.run()