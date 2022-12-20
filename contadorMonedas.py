import cv2
import numpy as np


def ordenarPuntos(puntos):
    n_puntos = np.concatenate(
        [puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
    y_order = sorted(n_puntos, key=lambda n_puntos: n_puntos[1])
    x1_order = y_order[:2]
    x1_order = sorted(x1_order, key=lambda x1_order: x1_order[0])
    x2_order = y_order[2:4]
    x2_order = sorted(x2_order, key=lambda x2_order: x2_order[0])

    return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]


def roi(image, ancho, alto):
    imagen_aliniada = None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    #cv2.imshow('th', th)
    # CAPTURA BLANCO Y NEGRO
    cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]

    for c in cnts:
        epsilon = 0.01*cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, epsilon, True)
        if len(aprox) == 4:
            puntos = ordenarPuntos(aprox)
            pts1 = np.float32(puntos)
            pts2 = np.float32([[0, 0], [ancho, 0], [0, alto], [ancho, alto]])
            m = cv2.getPerspectiveTransform(pts1, pts2)
            imagen_aliniada = cv2.warpPerspective(image, m, (ancho, alto))
    return imagen_aliniada


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    CUADRO = roi(frame, ancho=400, alto=600)
    if CUADRO is not None:
        puntos = []
        imagen_gris = cv2.cvtColor(CUADRO, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imagen_gris, (5, 5), 1)
        _, th_2 = cv2.threshold(
            blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        cv2.imshow('th_monedas', th_2)
        # MONEDAS EN BLANCO Y NEGRO
        cnts_2 = cv2.findContours(
            th_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        #cv2.drawContours(CUADRO, cnts_2, -1, (54, 227, 22), 2)
        S_0_10 = 0
        S_0_20 = 0
        S_0_50 = 0
        S_1_00 = 0
        S_2_00 = 0
        S_5_00 = 0
        for c_2 in cnts_2:
            area = cv2.contourArea(c_2)
            momentos = cv2.moments(c_2)

            # print(area)

            if (momentos["m00"] == 0):
                momentos["m00"] = 1.0

            x = int(momentos["m10"]/momentos["m00"])
            y = int(momentos["m01"]/momentos["m00"])

            if area < 5120 and area > 4800:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(CUADRO, "0.10", (x, y),
                            font, 0.70, (0, 0, 0), 2)
                S_0_10 = S_0_10 + 0.10

            if area < 6800 and area > 6300:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(CUADRO, "0.20", (x, y),
                            font, 0.70, (0, 0, 0), 2)
                S_0_20 = S_0_20 + 0.20

            if area < 7200 and area > 6900:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(CUADRO, "5.00", (x, y),
                            font, 0.70, (0, 0, 0), 2)
                S_5_00 = S_5_00 + 5.00

            if area < 8000 and area > 7400:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(CUADRO, "0.50", (x, y),
                            font, 0.70, (0, 0, 0), 2)
                S_0_50 = S_0_50 + 0.50

            if area < 10000 and area > 9400:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(CUADRO, "1.00", (x, y),
                            font, 0.70, (0, 0, 0), 2)
                S_1_00 = S_1_00 + 1.00

            if area < 11500 and area > 11000:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(CUADRO, "2.00", (x, y),
                            font, 0.70, (0, 0, 0), 2)
                S_2_00 = S_2_00 + 2.00

        TOTAL = S_0_10 + S_0_20 + S_0_50 + S_1_00 + S_2_00 + S_5_00
        print("TOTAL: ", round(TOTAL, 2))
        cv2.imshow('CONTADOR DE MONEDAS', CUADRO)
        #cv2.imshow('frame', frame)

    ecs = cv2.waitKey(1) & 0xFF

    if ecs == 27:
        break

cap.release()
cv2.destroyAllWindows()

