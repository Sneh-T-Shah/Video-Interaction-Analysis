from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace as df
import cv2
import time
import json

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def image_visuals(img):
    questions = ["what and where is the adult looking at in the image", "what and where is the child looking at", "what is the child doing in the image", "what is the adult doing in the image"]
    texts = ["Adult loking at: ","Child looking at: "]
    ans = []
    y_offset = 50  # Initial y-coordinate for the first question
    for i,question in enumerate(questions):
        prompt = f"Question: {question} Answer:"
        inputs = processor(img, text=prompt, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=10)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        ans.append(generated_text)
    ans[1] = ans[1][27:]
    for i in range(2):
        cv2.putText(img,texts[i]+": "+ans[i], (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        y_offset += 50  # Increment the y-coordinate for the next question
    return img,ans

def pipeline(video_path):
    t1 = time.time()
    cap = cv2.VideoCapture(video_path)
    fps = 1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/path_to_output.mp4', fourcc, fps, (width, height))

    analysis = {'child':{'frame':[],'query':[],'emotion':[]},'adult':{'frame':[],'query':[],'emotion':[]}}
    processed_frame = 0
    frame_cnt = 1
    emotion_person = {"child":{"happy":0,"neutral":0,"surprise":0},"adult":{"happy":0,"neutral":0,"surprise":0}}
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame_cnt%30 == 0:
                processed_frame+=1
                faces = df.analyze(frame, actions=['emotion','age'],enforce_detection=False,detector_backend='retinaface')
                ages = [face_info['age'] for face_info in faces]
                if not ages:
                    continue
                youngest_person_idx = ages.index(min(ages))
                if faces[0]['face_confidence']==1.0:
                    frame,ans = image_visuals(frame)

                for i,face in enumerate(faces):
                    if face['face_confidence']==1.0:
                        if len(faces)>1:
                            if i==youngest_person_idx:
                                person = "child"
                            else:
                                person = "adult"
                        else:
                            if ages[0]<25:
                                person = "child"
                            else:
                                person = "adult"

                        x,y,w,h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                        if face['emotion']['happy']>face['emotion']['neutral'] and face['emotion']['happy']>face['emotion']['surprise']:
                            emotion = "happy"
                        elif face['emotion']['neutral']>face['emotion']['happy'] and face['emotion']['neutral']>face['emotion']['surprise']:
                            emotion = "neutral"
                        else:
                            emotion = "surprise"

                        emotion_person[person][emotion] += 1
                        analysis[person]['frame'].append(processed_frame)
                        analysis[person]['emotion'].append(emotion)

                        if person == 'adult':
                            analysis['adult']['query'].append([ans[0],ans[3]])
                        else:
                            analysis['child']['query'].append([ans[1],ans[2]])

                        cv2.putText(frame, f"{person}: {emotion}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                out.write(frame)
            frame_cnt += 1
        else:
            break
    print("Video processing complete.")
    print("Output saved as output.mp4")
    print("Child emotions: ", emotion_person["child"])
    print("Adult emotions: ", emotion_person["adult"])
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    persons = ['child','adult']
    colors = ['red', 'blue', 'green']

    for person in persons:
        labels = emotion_person[person].keys()
        values = emotion_person[person].values()
        plt.bar(labels,values,color=colors)
        plt.xlabel("Emotions")
        plt.ylabel("No. of occurances: ")
        plt.savefig(f'{i} emotion.png')


    with open('child_data.json', 'w') as f:
        child_data = {
            'frame': analysis['child']['frame'],
            'query': analysis['child']['query'],
            'emotion': analysis['child']['emotion']
        }
        json.dump(child_data, f, indent=4)

    # Create the adult data file
    with open('adult_data.json', 'w') as f:
        adult_data = {
            'frame': analysis['adult']['frame'],
            'query': analysis['adult']['query'],
            'emotion': analysis['adult']['emotion']
        }
        json.dump(adult_data, f, indent=4)

    print("Time taken: ", time.time()-t1)
    print("time per frame: ", (time.time()-t1)/(frame_cnt//30))

video_path = 'path/to/video.mp4'
pipeline(video_path)
