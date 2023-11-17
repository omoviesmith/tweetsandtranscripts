# tweetsandtranscripts
## Flask Audio Transcription Service

A robust audio transcription service built with Flask, designed to extract tweets and transcribe audio from YouTube videos. Utilizing speaker diarization capabilities, it can accurately identify and differentiate between multiple speakers, particularly beneficial for podcasts or interviews. Coupled with AWS technologies and OpenAI's Whisper API for transcription, this application presents a dependable solution for generating transcriptions.

### Features

- **YouTube Audio Extraction**: Download audio from YouTube videos seamlessly and store it in an AWS S3 bucket for processing.
- **Speaker Diarization**: Determine distinct speakers within an audio file, which is vital for transcribing dialogues accurately.
- **Speech-to-Text**: Convert audio into text using the highly accurate OpenAI Whisper model API.
- **S3 Storage**: Store and manage audio files and transcription documents in Amazon S3 buckets securely.

### Technologies

- **Flask**: A lightweight WSGI web application framework.
- **Gunicorn**: A Python WSGI HTTP Server for UNIX, used to serve Flask applications in production.
- **Docker**: Containers used to create, deploy, and run applications by using containers.
- **AWS S3**: Scalable storage in the AWS cloud.
- **AWS Transcribe**: An AWS service that makes it easy for developers to add speech-to-text capability to applications.
- **OpenAI Whisper API**: State-of-the-art speech recognition model that provides highly accurate transcriptions.

### How It Works

1. **YouTube Audio Extraction**: Input the URL of a desired YouTube video, and this service will extract the audio content, storing it in a predetermined S3 bucket.

2. **Speaker Diarization**: Our service then performs speaker diarization to segment the audio into parts, each corresponding to a different speaker.

3. **Transcription**: The segmented audio is passed to the OpenAI Whisper API for transcription.

4. **Results Storage**: The final transcriptions are uploaded back into the S3 bucket for persistent storage and easy accessibility.

5. **CSV Generation**: The app can compile all transcriptions into a CSV file with corresponding timestamps, speaker labels, and produced text.

### Quick Start

1. Clone the repository
2. Install dependencies from `requirements.txt`
3. Build and run the Docker container
4. Access the service at `http://localhost:8000`

### Running the Application

Use the following Docker commands to build and run the application:

```
docker build -t flask-audio-transcription .
docker run -p 8000:8000 flask-audio-transcription
```

For a detailed guide on how to use this service, including API documentation, please refer to our wiki.

### Contribution

Interested in contributing? Great! Please read our CONTRIBUTING.md to learn about our development process, how to propose bugfixes and improvements, and how to build and test your changes to this project.

### Support and Feedback

If you have any feedback, issues, or questions about this application, please feel free to file an issue in this repository or contact us via our mailing list.

### License

This Flask Audio Transcription Service is released under the [MIT License](LICENSE).

