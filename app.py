from flask import Flask, render_template, request, redirect, url_for, session, Response
from flask_mysqldb import MySQL
from passlib.hash import sha256_crypt
import cv2
import mediapipe as mp
import numpy as np
import pickle

app = Flask(__name__)
app.secret_key = 'CA528EEFAC93C53B3391915E7AD8C'

# Configuración de MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'flask_login_example'

mysql = MySQL(app)

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.1)

labels_dict = {0: 'A', 1: 'B', 2: 'C'}

def gen_frames():
    cap = cv2.VideoCapture(0)
    # Configura la resolución de captura de la cámara web a un valor más bajo
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        data_aux = []
        x_ = []
        y_ = []

        success, frame = cap.read()
        if not success:
            break
        else:
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)
                        x_.append(x)
                        y_.append(y)

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = sha256_crypt.encrypt(request.form['password'])
        first_name = request.form['first_name']
        last_name = request.form['last_name']

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (username, password, first_name, last_name) VALUES (%s, %s, %s, %s)",
                    (username, password, first_name, last_name))
        mysql.connection.commit()
        cur.close()

        return redirect(url_for('login'))
    return render_template('register.html')
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password_candidate = request.form['password']

        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM users WHERE username = %s", [username])

        if result > 0:
            data = cur.fetchone()
            password = data[2]  # El índice 2 corresponde a la columna 'password'

            if sha256_crypt.verify(password_candidate, password):
                session['logged_in'] = True
                session['username'] = username
                return redirect(url_for('dashboard'))
            else:
                error = 'Invalid login'
                return render_template('login.html', error=error)
        else:
            error = 'Username not found'
            return render_template('login.html', error=error)

    return render_template('login.html')
        
@app.route('/dashboard')
def dashboard():
    if 'logged_in' in session:
        username = session['username']
        
        cur = mysql.connection.cursor()
        cur.execute("SELECT first_name, last_name FROM users WHERE username = %s", [username])
        user_data = cur.fetchone()
        name = user_data[0] + ' ' + user_data[1]
        
        return render_template('dashboard.html', name=name)
    else:
        return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/hand_gesture')
def hand_gesture():
    return render_template('hand_gesture.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
