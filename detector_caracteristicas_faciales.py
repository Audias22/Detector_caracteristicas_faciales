from deepface import DeepFace
import cv2
import threading

ultimo_resultado = None
ia_trabajando = False


def preprocesar(frame_copia):
    lab = cv2.cvtColor(frame_copia, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def analizar_fondo(frame_copia):
    global ultimo_resultado, ia_trabajando
    try:
        frame_mejorado = preprocesar(frame_copia)
        info_lista = DeepFace.analyze(frame_mejorado, actions=['age', 'gender', 'race', 'emotion'],
                                      enforce_detection=False)
        ultimo_resultado = info_lista[0]
    except:
        pass
    ia_trabajando = False


def texto(frame, txt, pos, escala=0.75):
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 4)
    cv2.putText(frame, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), 2)


cap = cv2.VideoCapture(0)

ani, ali = 0, 0
try:
    img = cv2.imread("img.png")
    img = cv2.resize(img, (0, 0), None, 0.22, 0.22)
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

        edad = info['age']
        emociones = info['dominant_emotion']
        race = info['dominant_race']
        confianza_raza = int(info['race'][race])

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

        color_rect = (255, 180, 0)
        if gen == 'Hombre':
            color_rect = (255, 100, 50)
        elif gen == 'Mujer':
            color_rect = (180, 100, 255)
        cv2.rectangle(frame, (xi, yi), (xf, yf), color_rect, 2)

        texto(frame, str(gen),                                     (80, 62))
        texto(frame, str(edad),                                    (92, 110))
        texto(frame, str(emociones),                               (92, 165))
        texto(frame, str(race) + ' ' + str(confianza_raza) + '%',  (92, 220))

    cv2.imshow("Deteccion de caracteristicas faciales", frame)

    t = cv2.waitKey(5)
    if t == 27:
        break

cv2.destroyAllWindows()
cap.release()
