from deepface import DeepFace
import cv2
import threading

ultimo_resultado = None
ia_trabajando = False


def analizar_fondo(frame_copia):
    global ultimo_resultado, ia_trabajando
    try:
        info_lista = DeepFace.analyze(frame_copia, actions=['age', 'gender', 'race', 'emotion'],
                                      enforce_detection=False)
        ultimo_resultado = info_lista[0]
    except:
        pass
    ia_trabajando = False


cap = cv2.VideoCapture(0)

try:
    img = cv2.imread("img.png")
    img = cv2.resize(img, (0, 0), None, 0.18, 0.18)
    ani, ali, c = img.shape
    tiene_logo = True
except Exception as e:
    print("Advertencia: No se encontró img.png o hubo un error al cargarla.")
    tiene_logo = False

print("Iniciando cámara. Presiona ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret: break

    if tiene_logo:
        try:
            frame[10:ani + 10, 10:ali + 10] = img
        except:
            pass

    if not ia_trabajando:
        ia_trabajando = True
        hilo = threading.Thread(target=analizar_fondo, args=(frame.copy(),), daemon=True)
        hilo.start()

    if ultimo_resultado is not None:
        info = ultimo_resultado

        region = info['region']
        xi, yi, w, h = region['x'], region['y'], region['w'], region['h']
        xf, yf = xi + w, yi + h

        cv2.rectangle(frame, (xi, yi), (xf, yf), (255, 255, 0), 2)

        edad = info['age']
        emociones = info['dominant_emotion']
        race = info['dominant_race']

        gen_dict = info['gender']
        if isinstance(gen_dict, dict):
            gen = max(gen_dict, key=gen_dict.get)
        else:
            gen = gen_dict

        if gen == 'Man':
            gen = 'Hombre'
            if emociones == 'angry': emociones = 'enojado'
            if emociones == 'disgust': emociones = 'disgustado'
            if emociones == 'fear': emociones = 'miedoso'
            if emociones == 'happy': emociones = 'feliz'
            if emociones == 'sad': emociones = 'triste'
            if emociones == 'surprise': emociones = 'sorprendido'
            if emociones == 'neutral': emociones = 'neutral'

            if race == 'asian': race = 'asiatico'
            if race == 'indian': race = 'indio'
            if race == 'black': race = 'negro'
            if race == 'white': race = 'blanco'
            if race == 'middle eastern': race = 'oriente medio'
            if race == 'latino hispanic': race = 'latino'

        elif gen == 'Woman':
            gen = 'Mujer'
            if emociones == 'angry': emociones = 'enojada'
            if emociones == 'disgust': emociones = 'disgustada'
            if emociones == 'fear': emociones = 'miedosa'
            if emociones == 'happy': emociones = 'feliz'
            if emociones == 'sad': emociones = 'triste'
            if emociones == 'surprise': emociones = 'sorprendida'
            if emociones == 'neutral': emociones = 'neutral'

            if race == 'asian': race = 'asiatica'
            if race == 'indian': race = 'india'
            if race == 'black': race = 'negra'
            if race == 'white': race = 'blanca'
            if race == 'middle eastern': race = 'oriente medio'
            if race == 'latino hispanic': race = 'latina'

        cv2.putText(frame, str(gen), (65, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, str(edad), (75, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, str(emociones), (75, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, str(race), (75, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Deteccion de caracteristicas faciales", frame)

    t = cv2.waitKey(5)
    if t == 27:
        break

cv2.destroyAllWindows()
cap.release()