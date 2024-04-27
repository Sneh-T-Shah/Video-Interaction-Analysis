# Video-Interaction-Analysis

This repository contains a powerful pipeline for analyzing interactions between individuals in videos. The pipeline leverages computer vision and natural language processing techniques to predict gaze direction, emotions, engagement levels, and object interactions, providing valuable insights for understanding human behavior and interactions in various domains, such as therapy sessions, educational settings, and more.

## Problem Statement

Understanding people's gaze, emotions, and object interactions in videos can provide valuable insights into human behavior and interactions. This project aims to develop an optimized inference pipeline that can analyze videos of interactions between individuals, such as therapy sessions, classroom settings, or any other scenario involving interpersonal interactions. The pipeline should predict the gaze direction, emotions, level of engagement, and object interactions for the individuals in the video.

## Solution Overview


To address this problem, I have implemented a multi-modal pipeline that integrates various computer vision and natural language processing models. The key components of the solution are:

1. **Visual Question Answering (VQA) Model**: I utilized the BLIP2 (Salesforce/blip2-opt-2.7b) VQA model from the Transformers library to predict the gaze direction, object interactions, and activity descriptions for the individuals in the video.

2. **Face Detection and Emotion Recognition**: For face detection and emotion recognition, I used the DeepFace library, specifically the RetinaFace model for face detection and the built-in emotion recognition method to classify emotions as happy, neutral, or surprise.

3. **Age Recognition**: To differentiate between the therapist (or teacher) and the person receiving therapy (or student), I implemented an age recognition algorithm. This allows the pipeline to accurately map the recognized emotions to the respective individuals based on their age.

4. **Video Processing Pipeline**: The pipeline includes steps for video preprocessing, face detection, emotion recognition, age recognition, VQA model integration, and output overlay. It generates an output video with bounding boxes around detected faces, emotion labels, gaze predictions, and bar plots showing emotion occurrences.

5. **Prompt Engineering**: To enhance the performance of the VQA model, I experimented with different prompts and templates, focusing on specific aspects and domain-specific terminology.

## Implementation Details

1. **Video Preprocessing**: To save computation time, the original 30 fps video is converted to 1 fps by skipping frames.

2. **Face Detection and Emotion Recognition**: For each frame, the DeepFace library is used to detect faces and recognize their emotions.

3. **Age Recognition**: An age recognition algorithm is implemented to differentiate between the therapist (or teacher) and the person receiving therapy (or student). This step ensures that the recognized emotions are accurately mapped to the respective individuals based on their age.

4. **VQA Model Integration**: The BLIP2 VQA model is integrated to generate answers to pre-defined questions about gaze direction, object interactions, and activity descriptions for the individuals in the video.

5. **Output Overlay**: The pipeline overlays the following information on the output video:
   - Bounding boxes around detected faces with person type (therapist/teacher or student/person receiving therapy) and detected emotion
   - Text indicating where the individuals are looking
   - Bar plots showing emotion occurrences for the individuals

6. **JSON Output**: Two JSON files are generated, one for the therapist (or teacher) and one for the person receiving therapy (or student), containing frame-by-frame information about detected emotions, gaze predictions, and activity descriptions.

## Running the Code

To run the code, you can simply download the provided Google Colab file (`video_interaction_analysis.ipynb`) and upload it to your Google Drive. Then, open the file in Google Colab and run it on a T4 GPU runtime. You'll need to update the input video paths in the code to point to your own video files. The code takes care of setting up the necessary dependencies and handles the entire pipeline, so you don't need to go through the trouble of setting up a virtual environment or managing dependencies manually.

## Results and Observations

The pipeline performed well, providing a rich set of information for analyzing interactions and behavior during various scenarios, such as therapy sessions, educational settings, or any other interpersonal interactions captured in videos.

## Limitations and Future Improvements

- Improve age detection accuracy for better mapping of faces to therapist/teacher and student/person receiving therapy.
- Access more computational resources to run the VQA model without quantization, potentially improving response quality.
- Continuous prompt optimization to enhance the VQA model's responses.

By addressing these limitations and leveraging advancements in machine learning and computer vision, the pipeline's performance and its ability to analyze interactions in various domains could be significantly improved.