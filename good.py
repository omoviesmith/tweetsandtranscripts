import torch
import audioread
import librosa
import re
import pandas as pd
from pydub import AudioSegment, silence
import youtube_dl
import yt_dlp as youtube_dl
import logging
from datetime import timedelta
import time
import tempfile
import boto3
# import youtube_dl
import csv
import os
import tempfile
from os import remove
import requests
import json
from dotenv import load_dotenv
from pprint import pprint
from yt_dlp.utils import DownloadError
from pyannote.audio import Pipeline
import torch
from pydub import AudioSegment
import numpy as np
import socket
import io
# import openai
# from youtube_dl.utils import DownloadError
import whisper
import concurrent.futures
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Tokenizer
from concurrent.futures import ThreadPoolExecutor
import openai
from openai import OpenAI

# AWS related variables
s3_bucket = 'twitsbucket'
folder = 'transcripts'

load_dotenv()  # take environment variables from .env.
bearer_token = os.getenv("BEARER_TOKEN")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# creating AWS clients
s3 = boto3.client('s3', 
                        aws_access_key_id=AWS_ACCESS_KEY_ID, 
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                        region_name=AWS_REGION)

transcribe = boto3.client('transcribe', 
                        aws_access_key_id=AWS_ACCESS_KEY_ID, 
                        aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                        region_name=AWS_REGION)

# Set default timeout for all sockets
socket.setdefaulttimeout(100)  # Time_in_seconds

default_bucket_name = "twittrans"


# method to extract audio from youtube video
def parse_s3_url(s3_url):
    # Assuming the S3 URL is in the standard format of 's3://bucket_name/object_key'
    # Extract the bucket name and object key from the S3 URL
    try:
        bucket_name, object_key = s3_url.replace("s3://", "").split('/', 1)
        return bucket_name, object_key
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
            print(f"Uploaded: Bucket name {bucket_name} and Object key {object_key}")

    return s3_url



def speaker_diarization(s3_url):
    bucket_name, object_key = parse_s3_url(s3_url)
    print(f"Downloaded: Bucket name {bucket_name} and Object key {object_key}")

    # Load the speaker diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0", use_auth_token="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")
    pipeline.to(torch.device("cpu"))

    # Initialize S3 client
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                      region_name=AWS_REGION)

    # Download the audio file as a streaming body object directly from S3
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    streaming_body = response['Body']

    # Use NamedTemporaryFile to create a temp file that PyAnnote can use, and convert to .wav if needed
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            # Write streaming body content to the temp file
            streaming_body_content = streaming_body.read()
            tmp_file.write(streaming_body_content)
            tmp_file.flush()
            tmp_file_path = tmp_file.name

        # Perform diarization using the temp file path
        diarization = pipeline(tmp_file_path)

        # Process diarization results
        speakers = set()
        result = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.add(speaker)
            result.append(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

    finally:
        # Clean up the temporary file
        if tmp_file_path is not None:
            os.remove(tmp_file_path)

    number_of_speakers = len(speakers)
    return result, number_of_speakers

# method to upload the files to s3 bucket
def upload_to_s3(file, input_pdf_file_path):
   
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION")
    
    s3_client = boto3.client('s3', 
                                aws_access_key_id=AWS_ACCESS_KEY_ID, 
                                aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                                region_name=AWS_REGION)
    
    default_bucket_name = "twittrans"
    filename = os.path.basename(input_pdf_file_path)
    key = 'audio/{}'.format(filename)
    
    try:
        s3_client.upload_fileobj(file, default_bucket_name, key)
        s3_url = f"s3://{default_bucket_name}/{key}"

        return s3_url

    except Exception as e:
        print(f"An error occurred: {e}")

def convert_str_list_to_timedelta(diarization_result):
    """
    Extract from Diarization result the given speakers with their respective speaking times 
    and transform them in pandas timedelta objects
    :param diarization_result: result of diarization as a list of strings
    :return: list with timedelta intervals and their respective speaker ID
    """

    diarization_timestamps = []
    for str_entry in diarization_result:
        # Use regex to extract start, stop and speaker id from string
        matches = re.match(r'start=(\d+\.\d+)s stop=(\d+\.\d+)s speaker_(\w+)', str_entry)
        start_time = pd.Timedelta(seconds=float(matches.group(1)))
        stop_time = pd.Timedelta(seconds=float(matches.group(2)))
        speaker_id = matches.group(3)

        diarization_timestamps.append([start_time, stop_time, speaker_id])
        
    return diarization_timestamps

def merge_speaker_times(diarization_timestamps, max_space):
    """
    Merge near times for each detected speaker (Same speaker during 1-2s and 3-4s -> Same speaker during 1-4s)
    :param diarization_timestamps: diarization list.
    :param max_space: Maximum temporal distance between two silences (in milliseconds).
    :return: list with timedelta intervals and their respective speaker.
    """
    threshold = pd.Timedelta(seconds=max_space / 1000)
    index = 0
    length = len(diarization_timestamps) - 1

    # Inspect each element in diarization_timestamps until reaching the second-last element.
    while index < length:
        # If the next interval is by the same speaker and occurs within the threshold after the current interval...
        if (diarization_timestamps[index + 1][2] == diarization_timestamps[index][2] and
            diarization_timestamps[index + 1][0] - threshold <= diarization_timestamps[index][1]):

            # Extend the current interval to also include the next interval, then delete the next interval.
            diarization_timestamps[index][1] = diarization_timestamps[index + 1][1]
            del diarization_timestamps[index + 1]
            length -= 1
        else:
            index += 1

    return diarization_timestamps

def extending_timestamps(new_diarization_timestamps):
    """
    Extend timestamps between each diarization timestamp if possible, so we avoid word cutting
    :param new_diarization_timestamps: list
    :return: list with merged times
    """

    for i in range(1, len(new_diarization_timestamps)):
        if new_diarization_timestamps[i][0] - new_diarization_timestamps[i - 1][1] <= timedelta(milliseconds=3000) and new_diarization_timestamps[i][0] - new_diarization_timestamps[i - 1][1] >= timedelta(milliseconds=100):
            middle = (new_diarization_timestamps[i][0] - new_diarization_timestamps[i - 1][1]) / 2
            new_diarization_timestamps[i][0] -= middle
            new_diarization_timestamps[i - 1][1] += middle

    # Converting list so we have a milliseconds format
    for elt in new_diarization_timestamps:
        elt[0] = elt[0].total_seconds() * 1000
        elt[1] = elt[1].total_seconds() * 1000

    return new_diarization_timestamps

def transcribe_audio_part(filename, stt_model, stt_tokenizer, myaudio_path, sub_start, sub_end, index, to_end=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        with torch.no_grad():
            # Load audio with pydub
            myaudio = AudioSegment.from_file(myaudio_path)

            if to_end:
                new_audio = myaudio[sub_start * 1000:]  # till the end of the audio
            else:
                new_audio = myaudio[sub_start * 1000 : sub_end * 1000]  # Works in milliseconds
                        # Create "sound" directory if it doesn't exist
            if not os.path.exists("sound"):
                os.makedirs("sound")

            path = os.path.join("sound", filename[:-4] + "_audio_" + str(index) + ".mp3")
            # path = filename[:-4] + "_audio_" + str(index) + ".wav"  # Given filename should end with ".wav"
            new_audio.export(path)  # Exports to a wav file in the current path.

            # Load audio file with librosa, set sound rate to 16000 Hz because the model we use was trained on 16000 Hz data
            input_audio, _ = librosa.load(path, sr=16000)

            # return PyTorch torch.Tensor instead of a list of python integers thanks to return_tensors = ‘pt’
            input_values = stt_tokenizer(input_audio, return_tensors="pt").input_values

            # Get logits from the data structure containing all the information returned by the model and get our prediction
            logits = stt_model.to(device)(input_values).logits
            prediction = torch.argmax(logits, dim=-1)

            # Decode & lower our string (model's output is only uppercase)
            if isinstance(stt_tokenizer, Wav2Vec2Tokenizer):
                transcription = stt_tokenizer.batch_decode(prediction)[0]
            elif isinstance(stt_tokenizer, Wav2Vec2Processor):
                transcription = stt_tokenizer.decode(prediction[0])

            # return transcription
            return transcription.lower()

    except Exception as e:
        print(f"Error occurred while processing audio: {e}")


# def transcribe_audio_whisper(job_name, s3_url, start=None, end=None, speaker_label=None):
#     # Initialize OpenAI client
#     # openai.api_key = 'your-api-key'  # Replace with your OpenAI API key.
#     openai.api_key = OPENAI_API_KEY
#     # Set up S3 client
#     # s3_client = boto3.client('s3')
#     bucket_name, object_key = parse_s3_url(s3_url)
    
#     # Download file from S3 to in-memory bytes buffer
#     audio_buffer = io.BytesIO()
#     s3.download_fileobj(bucket_name, object_key, audio_buffer)
#     audio_buffer.seek(0)

    
    
#     # Load the audio segment using pydub if start and end are specified, otherwise use the whole audio
#     if start is not None and end is not None:
#         audio = AudioSegment.from_file(audio_buffer, format="mp3")[start*1000:end*1000]  # Adjust format if needed.
#     else:
#         audio = AudioSegment.from_file(audio_buffer, format="mp3")


#     # Prepare a buffer for the Whisper API
#     whisper_buffer = io.BytesIO()
#     whisper_buffer.name = object_key  # Set name with the appropriate file extension (e.g. '.wav' or '.wav').

#     # Export the audio segment to the buffer
#     audio.export(whisper_buffer, format="mp3")
#     whisper_buffer.seek(0)
    
#     # Transcribe the audio using the Whisper API
#     transcript = openai.Audio.transcribe("whisper-1", whisper_buffer)
    
#     # Close the buffer
#     whisper_buffer.close()
#     # Include speaker label in the transcription
#     if speaker_label is not None:
        
#         return f"{speaker_label}: {transcript['text']}"
#         print(f"{speaker_label}: {transcript['text']}")
#     else:
#         return transcript['text']

# def transcribe_audio_whisper(job_name, s3_url, start=None, end=None, speaker_label=None):
#     # Set the API key for the OpenAI client
#     openai.api_key = OPENAI_API_KEY  # Set this in your environment variables or configuration
#     # client=OpenAI
#     # Parse the S3 URL to get the bucket and key
#     bucket_name, object_key = parse_s3_url(s3_url)

#     # # Set up S3 client
#     # s3_client = boto3.client('s3')

#     # Download file from S3 to in-memory bytes buffer
#     audio_buffer = io.BytesIO()
#     s3.download_fileobj(bucket_name, object_key, audio_buffer)
#     audio_buffer.seek(0)

#     # Load and optionally trim the audio segment using pydub
#     if start is not None and end is not None:
#         audio_segment = AudioSegment.from_file(io.BytesIO(audio_buffer.read()), format="mp3")[start*1000:end*1000]
#     else:
#         audio_segment = AudioSegment.from_file(io.BytesIO(audio_buffer.read()), format="mp3")

#     # Transcription using OpenAI client; create a new in-memory bytes buffer for trimmed audio
#     trimmed_audio_buffer = io.BytesIO()
#     audio_segment.export(trimmed_audio_buffer, format="mp3")
#     trimmed_audio_buffer.seek(0)

#     # Clean the presigned URL to exclude the query parameters (Not recommended as it breaks access)
#     # clean_presigned_url = presigned_url.split('?')[0]
#     # print(clean_presigned_url)
#     # Transcribe the audio using OpenAI's Whisper model
#     # openai.audio.transcriptions.create
#     transcript_response = openai.audio.transcriptions.create(
#         file=trimmed_audio_buffer,
#         model="whisper-1",
#         response_format="text"
#         # You might need to pass any additional parameters as required by the API
#     )
    
#     # Extract the transcription text
#     transcription_text = transcript_response['text']
    
#     # Include speaker label in the transcription if provided
#     if speaker_label is not None:
#         transcription_text = f"{speaker_label}: {transcription_text}"

#     # Clean up: delete the trimmed file from S3 if it was created
#     # if start is not None and end is not None:
#     #     s3.delete_object(Bucket=bucket_name, Key=trimmed_key)
        
#     return transcription_text

def transcribe_audio_whisper(job_name, s3_url, start=None, end=None, speaker_label=None):
    # ...

    client = OpenAI()
    bucket_name, object_key = parse_s3_url(s3_url)

    # Download file from S3 to in-memory bytes buffer
    audio_buffer = io.BytesIO()
    s3.download_fileobj(bucket_name, object_key, audio_buffer)
    audio_buffer.seek(0)

    # Load and optionally trim the audio segment using pydub
    if start is not None and end is not None:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_buffer.read()), format="mp3")[start*1000:end*1000]
    else:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_buffer.read()), format="mp3")

    # Create a temporary directory to store the audio
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define audio file path
        temp_file_path = os.path.join(temp_dir, 'temp_audio.mp3')

        # Export audio to temporary file
        audio_segment.export(temp_file_path, format='mp3')

        # Transcribe the audio using OpenAI's Whisper model
        with open(temp_file_path, 'rb') as audio_file:
            transcript_response = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="text"
            )

        # File and the directory will be automatically deleted upon leaving the `with` block

    # Extract the transcription text
    print(transcript_response)
    transcription_text = transcript_response
    
    return transcription_text

# def parse_s3_url(s3_url):
#     # Extracts the bucket name and object key from the S3 URL
#     # Assuming the S3 URL format is "s3://bucket-name/object-key"
#     if not s3_url.startswith("s3://"):
#         raise ValueError("Invalid S3 URL format.")
#     parts = s3_url[5:].split("/", 1)
#     return parts[0], parts[1]

def transcribe_audio(filename, job_name, myaudio_path, sub_start, sub_end, index, to_end=False):
    myaudio = AudioSegment.from_file(myaudio_path)

    if to_end:
        new_audio = myaudio[sub_start * 1000:]  # till the end of the audio
    else:
        new_audio = myaudio[sub_start * 1000 : sub_end * 1000]  # Works in milliseconds

    # Export audio segment to an in-memory byte stream and create a streaming body object for boto3
    byte_stream = io.BytesIO()
    new_audio.export(byte_stream, format="mp3")
    byte_stream.seek(0)  # Important: reset stream position to the beginning before uploading
    # file_obj = boto3.StreamingBody(byte_stream, len(byte_stream.getvalue()))

    # Define path/key in S3
    path = filename[:-4] + "_audio_" + str(index) + ".mp3"
    print(f"Running transcribe audio function: ", {path})
    # Upload the streaming body object directly to S3 without saving locally
    s3 = boto3.client('s3', 
                    aws_access_key_id=AWS_ACCESS_KEY_ID, 
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY, 
                    region_name=AWS_REGION)
    s3.upload_fileobj(byte_stream, s3_bucket, f'{folder}/{path}')

    # Transcribe
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': f's3://{s3_bucket}/{folder}/{path}'},
        LanguageCode='en-US'
    )

    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            if status['TranscriptionJob']['TranscriptionJobStatus'] == 'FAILED':
                print("Transcription failed with reason: ", status['TranscriptionJob']['FailureReason'])
            else:
                transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            time.sleep(5)
            break
        print("Not ready yet...")
        # print(status)


    # Get the transcription result
    transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri'] 
    transcript_json = requests.get(transcript_uri).json()

    # Lowercase the transcription result and return it
    return transcript_json['results']['transcripts'][0]['transcript'].lower()


def get_parts_from_diarization(diarization_result, max_space=500):
    # Convert string diarization results to time intervals with timedelta objects
    timestamp_results = convert_str_list_to_timedelta(diarization_result)

    # Merge intervals from the same speaker if they are within the specified threshold
    merged_timestamps = merge_speaker_times(timestamp_results, max_space)

    # Extend timestamps to avoid potential cutoffs of words
    extend_timestamps = extending_timestamps(merged_timestamps)

    # Convert extended timestamp intervals to (start, end, speaker) parts in seconds for transcription
    parts = [(int(start / 1000), int(end / 1000), speaker) for (start, end, speaker) in extend_timestamps]
    print(parts)
    return parts


def transcription_diarization(filename, diarization_timestamps, myaudio_path, job_name):
    result = []
    for index, (sub_start, sub_end, speaker) in enumerate(diarization_timestamps):
        # Check if audio duration >= 0.5 ms
        audio_duration_ms = sub_end - sub_start
        if audio_duration_ms >= 500:
            transcription = transcribe_audio(filename, job_name+str(index), myaudio_path, sub_start/1000, sub_end/1000, index)
            result.append(f"{speaker}: {transcription}")  
        else:
            print(f"Skipping small audio segment from {sub_start} to {sub_end}. Duration: {audio_duration_ms} ms")
    return "\n".join(result)



# def parallel_transcribe_whisper(job_name, s3_audio_url, diarization_timestamps):
#     def transcribe_part(part):
#         start, end, speaker_label = part
#         audio_duration_seconds = (end - start)
#         if audio_duration_seconds > 0.1:
#             return transcribe_audio_whisper(job_name, s3_audio_url, start, end, speaker_label)
#         else:
#             print(f"Audio Segment lesser than: ", {audio_duration_seconds})
    
#     transcriptions = []
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = [executor.submit(transcribe_part, part) for part in diarization_timestamps]
#         for future in concurrent.futures.as_completed(futures):
#             transcriptions.append(future.result())
    
#     combined_transcription = "\n".join(transcriptions)
#     print(combined_transcription)
#     return combined_transcription

def parallel_transcribe_whisper(job_name, s3_audio_url, diarization_timestamps):
    def transcribe_part(part):
        start, end, speaker_label = part
        audio_duration_seconds = (end - start)
        if audio_duration_seconds >= 0.1:  # Use >= to include segments precisely at 0.1 seconds
            try:
                # Attempt transcription and return the result
                return transcribe_audio_whisper(job_name, s3_audio_url, start, end, speaker_label)
            except Exception as e:
                # Return an error message instead of None if transcription fails
                return f"Error transcribing segment {start}-{end} for speaker {speaker_label}: {e}"
        else:
            # Notify if the audio segment is too short
            print(f"Audio Segment lesser than: ", {audio_duration_seconds})
            # Return a message or empty string instead of None
            return f""

    transcriptions = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(transcribe_part, part) for part in diarization_timestamps]
        # Use a list comprehension to exclude None results
        transcriptions = [future.result() for future in concurrent.futures.as_completed(futures) if future.result() is not None]
    
    # Now this join should not fail as None results have been excluded from the list
    combined_transcription = "\n".join(transcriptions)

    return combined_transcription



def process_audio(url, diarization):
    # Create a job name using the video ID from the URL
    video_id = url.split('/')[-1]  # Assuming the video ID is the last part of the URL

    # Job name to be used consistently across the transcription process
    job_name = 'transcription_job_' + video_id

    # Extract audio from YouTube video and upload to S3
    audio_s3_url = extract_audio_from_yt_video(url)
    print(audio_s3_url)

    if diarization == False:
        print("No diarization needed!")
        transcription = transcribe_audio_whisper(job_name, audio_s3_url)
    # Transcription process based on the number of speakers detected
    elif number_of_speakers == 1:
        # Perform speaker diarization
        diarization_result, number_of_speakers = speaker_diarization(audio_s3_url)
        # If there's only one speaker, perform full transcription without diarization
        print("There's only one speaker!")
        transcription = transcribe_audio_whisper(job_name, audio_s3_url)
    else:
        # If there are multiple speakers, get the time parts for diarization and transcribe
        print("Diarization will be done for multiple speakers!")
        parts = get_parts_from_diarization(diarization_result)
        transcription = parallel_transcribe_whisper(job_name, audio_s3_url, parts)
        
    return transcription

def extract_transcription_from_yt_video(url):
    try:
        # Options for downloading subtitles
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],  # Specify the desired subtitle language
            'quit': True  # Exit after downloading subtitles without throwing an error code
        }

        # Initializing list to hold subtitles text
        subtitles_list = []

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            # Download video info
            info = ydl.extract_info(url, download=False)
            
            # Check if subtitles are available
            if 'requested_subtitles' in info and info['requested_subtitles'] and \
               'en' in info['requested_subtitles']:
                subtitle_url = info['requested_subtitles']['en']['url']
                # Download subtitles file
                subtitles_text = ydl.download([subtitle_url])
                # Append subtitle text to list
                subtitles_list.append(subtitles_text)
            else:
                print("No English subtitles available for this video.")
                return None

        # Combine subtitles into single text
        full_transcription = "\n".join(subtitles_list)
        return full_transcription

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
