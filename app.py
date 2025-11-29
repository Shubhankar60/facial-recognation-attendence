from flask import redirect, url_for
from flask import send_file
import cv2
import os
from flask import Flask, request, render_template, Response, redirect, url_for
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__)

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv','w') as f:
        f.write('Name,Roll,Time')


def totalreg():
    return len(os.listdir('static/faces'))


def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return face_detector.detectMultiScale(gray, 1.3, 5)


def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


def train_model():
    faces = []
    labels = []
    for user in os.listdir('static/faces'):
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)

    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    return df['Name'], df['Roll'], df['Time'], len(df)


def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    time_now = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv','a') as f:
            f.write(f'\n{username},{userid},{time_now}')


# GLOBAL STOP FLAGS
stop_attendance_flag = False
stop_register_flag = False


# ----------- STREAM FOR ATTENDANCE ----------
def gen_frames_attendance():
    global stop_attendance_flag
    stop_attendance_flag = False
    cap = cv2.VideoCapture(0)

    while not stop_attendance_flag:
        success, frame = cap.read()
        if not success:
            break

        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            try:
                person = identify_face(face.reshape(1, -1))[0]
                add_attendance(person)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,20), 2)
                cv2.putText(frame, person, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,20),2)
            except:
                pass

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/video_feed_attendance')
def video_feed_attendance():
    return Response(gen_frames_attendance(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_attendance')
def stop_attendance():
    global stop_attendance_flag
    stop_attendance_flag = True
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times,
                           l=l, totalreg=totalreg(), datetoday2=datetoday2)
@app.route('/download_attendance')
def download_attendance():
    filename = f"Attendance/Attendance-{datetoday}.csv"
    return send_file(filename, as_attachment=True)

@app.route('/reset_attendance')
def reset_attendance():
    filename = f"Attendance/Attendance-{datetoday}.csv"
    with open(filename, "w") as f:
        f.write("Name,Roll,Time")
    return redirect(url_for('home'))



# ----------- STREAM FOR REGISTRATION ----------
global_reg_username = ""
global_reg_userid = ""


def gen_frames_register():
    global stop_register_flag
    stop_register_flag = False
    cap = cv2.VideoCapture(0)

    sampleNum = 0
    j = 0
    folder = f"static/faces/{global_reg_username}_{global_reg_userid}"

    while not stop_register_flag and sampleNum < 50:
        success, frame = cap.read()
        if not success:
            break

        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            if j % 10 == 0:
                img_name = f"{global_reg_username}_{sampleNum}.jpg"
                cv2.imwrite(f"{folder}/{img_name}", frame[y:y+h, x:x+w])
                sampleNum += 1

            j += 1

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    train_model()


@app.route('/video_feed_register')
def video_feed_register():
    return Response(gen_frames_register(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_register')
def stop_register():
    global stop_register_flag
    stop_register_flag = True
    return redirect(url_for('home'))


# ---------------- ROUTES ----------------
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls,
                           times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/start')
def start():
    return render_template('start.html', datetoday2=datetoday2)


@app.route('/register_cam', methods=['POST'])
def register_cam():
    global global_reg_username, global_reg_userid
    global_reg_username = request.form['newusername']
    global_reg_userid = request.form['newuserid']

    folder = f"static/faces/{global_reg_username}_{global_reg_userid}"
    if not os.path.isdir(folder):
        os.makedirs(folder)

    return render_template('register.html', username=global_reg_username,
                           userid=global_reg_userid, datetoday2=datetoday2)


if __name__ == '__main__':
    app.run(debug=True)
