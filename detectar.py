from pathlib import Path
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

def analizar(ruta, conf=0.5):
    r = model(ruta, conf=conf, verbose=False)[0]
    alicates, tipos = [], []
    for b in r.boxes:
        nombre = r.names[int(b.cls[0])]
        caja = tuple(map(int, b.xyxy[0]))
        item = (nombre, float(b.conf[0]), caja)
        (alicates if nombre == "Plier" else tipos).append(item)

    resultado = []
    for _, c, (x1, y1, x2, y2) in alicates:
        tipo = None
        for nombre, ct, (tx1, ty1, tx2, ty2) in tipos:
            cx, cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                if tipo is None or ct > tipo[1]:
                    tipo = (nombre, ct)
        resultado.append({"conf": c, "tipo": tipo})
    return r, resultado

# Imagen de prueba: la primera del set de test
ruta = next(Path("pliers_dataset/test/images").glob("*.jpg"))
print("Probando con:", ruta.name)

r, alicates = analizar(str(ruta))
print(f"Alicates encontrados: {len(alicates)}")
for a in alicates:
    print(a)

cv2.imshow("Leyca", r.plot())
cv2.waitKey(0)
cv2.destroyAllWindows()