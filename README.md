# Leyca Vision: reconocimiento de alicates con YOLO

Módulo de visión artificial del **Proyecto Leyca** (gestión de insumos informáticos e inventario). Detecta alicates en una imagen, los cuenta y clasifica su tipo. Está pensado para integrarse con la aplicación principal y alimentar el inventario.

> **Estado actual:** prototipo funcional entrenado con un dataset público. Falta validarlo con fotos de las herramientas reales de Leyca (ver [Pendientes](#pendientes-y-siguientes-pasos)).

---

## Contenido

1. [Resumen](#resumen)
2. [Por qué YOLO y no Haar Cascade](#por-qué-yolo-y-no-haar-cascade)
3. [Dataset y atribución](#dataset-y-atribución)
4. [Cómo está etiquetado: dos cajas por alicate](#cómo-está-etiquetado-dos-cajas-por-alicate)
5. [Estructura del proyecto](#estructura-del-proyecto)
6. [Instalación y uso local](#instalación-y-uso-local)
7. [Entrenamiento en Google Colab](#entrenamiento-en-google-colab)
8. [Resultados](#resultados)
9. [Cómo integrarlo con la aplicación principal](#cómo-integrarlo-con-la-aplicación-principal)
10. [Problemas frecuentes](#problemas-frecuentes)
11. [Pendientes y siguientes pasos](#pendientes-y-siguientes-pasos)

---

## Resumen

| Elemento | Detalle |
|---|---|
| Tarea | Detección de objetos |
| Modelo | YOLOv8n (preentrenado, ajustado con *fine-tuning*) |
| Librería | Ultralytics |
| Clases | `Combination`, `Diagonal`, `Longnose`, `Plier`, `Slip Joint` |
| Entrenamiento | Google Colab, GPU T4, 50 épocas, imagen de 640 px, batch 16 |
| Modelo entrenado | `best.pt` (en la raíz del repo, unos 6 MB) |
| Inferencia | Funciona en CPU, no requiere GPU |

---

## Por qué YOLO y no Haar Cascade

El archivo inicial (`LEICA_CVIA.PY`) probaba Haar Cascade con los clasificadores de rostro y ojos de OpenCV. Se descartó para herramientas porque:

- No existen cascadas preentrenadas para herramientas; habría que entrenar una por tipo.
- Es muy sensible a rotación, iluminación, fondo y escala.
- Un clasificador por clase escala mal.
- Es una técnica de 2001; YOLO detecta y clasifica varias clases a la vez, con mucha mejor precisión.

---

## Dataset y atribución

Se usó un dataset público de Roboflow Universe:

- **Nombre:** Pliers Computer Vision Model
- **Autor:** LBYMF2H Pliers
- **Versión:** 9 (Roboflow Instant 2 [Eval])
- **Licencia:** CC BY 4.0
- **URL:** https://universe.roboflow.com/lbymf2h-pliers/pliers-s1ike/dataset/9

> **Importante:** la licencia CC BY 4.0 exige **dar crédito al autor**. Esta sección cumple ese requisito; no debe eliminarse, y debe mantenerse en cualquier documentación o entrega que use el modelo.

Contenido del dataset:

| Partición | Imágenes |
|---|---|
| `train` | 2.880 |
| `valid` | 120 |
| `test` | 120 |
| **Total** | **3.120** |

Las imágenes de `train` incluyen copias aumentadas (giros, brillo, etc.) generadas por Roboflow, por lo que la variedad real es menor a lo que sugiere la cifra. `valid` y `test` son imágenes originales.

**El dataset NO está en este repositorio** (pesa demasiado y está excluido en `.gitignore`). Hay que descargarlo de Roboflow (ver [Instalación](#instalación-y-uso-local)).

---

## Cómo está etiquetado: dos cajas por alicate

Este punto es clave para quien integre el módulo.

Cada imagen del dataset trae **dos cajas**:

1. Una caja de clase **`Plier`** que abarca el alicate completo.
2. Una caja de **tipo** (`Combination`, `Diagonal`, `Longnose` o `Slip Joint`) sobre la cabeza/punta del alicate.

Evidencia: en el conjunto `test` hay 120 imágenes con 240 instancias, de las cuales 120 son `Plier` y 30 de cada tipo.

**Consecuencia:** para **contar alicates** hay que contar únicamente las cajas `Plier`. Contar todas las cajas duplicaría el resultado. La función `analizar()` de `detectar.py` ya lo hace: cuenta las cajas `Plier` y le asigna a cada una el tipo cuya caja tenga el centro dentro de ella.

---

## Estructura del proyecto

```
leyca_vision/
├── best.pt              # modelo entrenado
├── detectar.py          # inferencia sobre una imagen
├── entrenar.py          # entrenamiento (pensado para correr en Colab)
├── LEICA_CVIA.PY        # prueba inicial con Haar Cascade (descartada)
├── .gitignore
├── README.md
├── dataset/             # reservada para fotos propias (excluida de git)
└── pliers_dataset/      # dataset de Roboflow (excluido de git)
    ├── train/  (images/, labels/)
    ├── valid/  (images/, labels/)
    ├── test/   (images/, labels/)
    └── data.yaml
```

`.gitignore` excluye `pliers_dataset/`, `dataset/`, `runs/`, `__pycache__/`, `.venv/` y `.vscode/`.

---

## Instalación y uso local

### Requisitos

- Python 3.10 a 3.13 recomendado. (El desarrollo se hizo en un equipo con Python 3.14; si `torch` no se instala, usar 3.12 o 3.13.)
- Git.

### Pasos

```powershell
git clone https://github.com/mrdoombyte/leyca_vision.git
cd leyca_vision
python -m pip install ultralytics opencv-python
```

> En Windows se usa `python -m pip` porque `pip` a secas puede no estar en el PATH. Al instalar Python, marcar **"Add Python to PATH"**.

### Descargar el dataset (solo para reentrenar)

1. Entrar a la URL del dataset (arriba) → **Download Dataset**.
2. Formato **YOLOv8** (o YOLOv11) → *Download zip to computer*.
3. Descomprimir dentro de `leyca_vision/` y renombrar la carpeta a `pliers_dataset`.
4. Revisar que `data.yaml` quede con rutas correctas, por ejemplo:

```yaml
path: pliers_dataset
train: train/images
val: valid/images
test: test/images

nc: 5
names: ['Combination', 'Diagonal', 'Longnose', 'Plier', 'Slip Joint']
```

Para solo **usar** el modelo (detectar) no hace falta el dataset: basta `best.pt`.

### Probar la detección

```powershell
python detectar.py
```

Por defecto toma una imagen de `pliers_dataset/test/images`, imprime cuántos alicates encontró y su tipo, y abre una ventana con las cajas dibujadas. Para usar una foto propia, cambiar la línea `ruta = ...` por `ruta = "mi_foto.jpg"`.

Salida esperada (ejemplo real):

```
Alicates encontrados: 1
{'conf': 0.8255, 'tipo': ('Longnose', 0.7716)}
```

> Ejecutar siempre los comandos desde la carpeta del proyecto, para que las rutas relativas se resuelvan bien.

---

## Entrenamiento en Google Colab

El equipo de desarrollo no tenía GPU NVIDIA (`torch.cuda.is_available()` devolvía `False`), así que el entrenamiento se hizo en Colab (GPU T4 gratuita). Cada época tarda unos 40 segundos; las 50 épocas, unos 35 minutos.

### Preparación

1. Comprimir `pliers_dataset/` en `pliers_dataset.zip` y subirlo a Google Drive, en una carpeta `Leyca`.
2. En Colab: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*.

### Celdas

**1. Instalar y montar Drive**

```python
!pip install ultralytics -q
from google.colab import drive
drive.mount('/content/drive')
```

**2. Descomprimir en el disco de Colab**

El zip **no trae carpeta contenedora**, por eso se extrae con `-d` a una carpeta propia:

```python
!rm -rf /content/train /content/valid /content/test /content/pliers_dataset
!mkdir -p /content/pliers_dataset
!unzip -q -o "/content/drive/MyDrive/Leyca/pliers_dataset.zip" -d /content/pliers_dataset
!ls /content/pliers_dataset
```

Debe listar `test  train  valid`.

**3. Escribir el `data.yaml` con rutas de Colab**

```python
yaml_text = """path: /content/pliers_dataset
train: train/images
val: valid/images
test: test/images

nc: 5
names: ['Combination', 'Diagonal', 'Longnose', 'Plier', 'Slip Joint']
"""
with open('/content/pliers_dataset/data.yaml', 'w') as f:
    f.write(yaml_text)
```

**4. Entrenar**

```python
import torch
print("GPU:", torch.cuda.is_available())   # debe imprimir True

from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(
    data="/content/pliers_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="alicates_v1",
    project="/content/drive/MyDrive/Leyca/runs",  # guarda en Drive
)
```

Los resultados se guardan en Drive para no perderlos si Colab se desconecta. Si se corta, se puede retomar con `YOLO(".../weights/last.pt")` y `model.train(resume=True)`.

**5. Evaluar sobre `test`**

```python
model = YOLO("/content/drive/MyDrive/Leyca/runs/alicates_v1/weights/best.pt")
metrics = model.val(data="/content/pliers_dataset/data.yaml", split="test")
```

Luego se descarga `best.pt` desde `Leyca/runs/alicates_v1/weights/` y se coloca en la raíz del proyecto.

> **Ojo con Linux y las mayúsculas:** en Colab, `Leyca` y `leyca` son carpetas distintas. Un error de mayúsculas fue el origen del primer fallo de `unzip`.

---

## Resultados

Evaluación sobre el conjunto `test` (120 imágenes que el modelo no vio durante el entrenamiento):

| Clase | Instancias | Precisión | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| **Todas** | 240 | 0,984 | 0,993 | 0,994 | 0,686 |
| Combination | 30 | 0,999 | 1,000 | 0,995 | 0,652 |
| Diagonal | 30 | 0,968 | 0,967 | 0,992 | 0,688 |
| Longnose | 30 | 1,000 | 0,998 | 0,995 | 0,660 |
| Plier | 120 | 0,990 | 1,000 | 0,995 | 0,758 |
| Slip Joint | 30 | 0,962 | 1,000 | 0,995 | 0,673 |

Velocidad en la GPU T4: unos 4,5 ms de inferencia por imagen. En CPU será más lenta (decenas de ms a un par de segundos), suficiente para registrar inventario.

### Cómo leer estos números con cuidado

- `test` proviene **del mismo dataset** que el entrenamiento, con fondos y cámaras parecidos. Un mAP50 de 0,99 es esperable ahí y **no garantiza** el mismo rendimiento con fotos reales de Leyca.
- Hay solo 30 ejemplos por tipo; una única equivocación mueve la métrica por clase unos 3 puntos.
- En una primera prueba manual, el detector acertó el tipo pero con confianza moderada (`Plier` 0,83, `Longnose` 0,77), inferior a lo que sugerían las métricas.
- La validación definitiva es con fotos propias (ver pendientes).

---

## Cómo integrarlo con la aplicación principal

> Esta sección es una **propuesta**. Todavía no hay código de integración en el repo, y se desconoce el stack de la app principal. Se ajusta según cómo esté hecha.

### Idea general

El módulo de visión funciona como un **servicio independiente**: recibe una imagen y devuelve un JSON. La app principal solo consume ese JSON y actualiza el inventario.

```
[Usuario sube foto] → [App principal] → POST /detectar → [Servicio de visión (YOLO)]
                                                              │
[Inventario / BD] ← [App principal] ←──── JSON con tipos y conteo
```

Separarlo así permite que el equipo de la app no tenga que instalar ni entender PyTorch: solo llama a un endpoint.

### Contrato propuesto del endpoint

**Petición:** `POST /detectar` con la imagen como `multipart/form-data` (campo `file`).

**Respuesta:**

```json
{
  "total_alicates": 2,
  "alicates": [
    {
      "tipo": "Longnose",
      "confianza_tipo": 0.77,
      "confianza_alicate": 0.83,
      "caja": [183, 135, 655, 380]
    },
    {
      "tipo": null,
      "confianza_tipo": null,
      "confianza_alicate": 0.61,
      "caja": [40, 52, 300, 210]
    }
  ]
}
```

`tipo` puede ser `null` si se detectó el alicate pero no se pudo asignar un tipo; el consumidor debe contemplarlo.

### Ejemplo de implementación con FastAPI

Instalación: `python -m pip install fastapi uvicorn python-multipart`.

Archivo `servicio.py`:

```python
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO

app = FastAPI(title="Leyca Vision")

# El modelo se carga UNA SOLA VEZ al arrancar el servicio.
model = YOLO("best.pt")
CONF_MIN = 0.5


def analizar(img, conf=CONF_MIN):
    r = model(img, conf=conf, verbose=False)[0]
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
        resultado.append({
            "tipo": tipo[0] if tipo else None,
            "confianza_tipo": round(tipo[1], 3) if tipo else None,
            "confianza_alicate": round(c, 3),
            "caja": [x1, y1, x2, y2],
        })
    return resultado


@app.post("/detectar")
async def detectar(file: UploadFile = File(...)):
    datos = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Imagen no válida")
    alicates = analizar(img)
    return {"total_alicates": len(alicates), "alicates": alicates}
```

Ejecutar:

```powershell
uvicorn servicio:app --reload --port 8000
```

La documentación interactiva queda en `http://localhost:8000/docs`, donde se puede probar subiendo una foto.

### Reglas importantes para la integración

1. **Cargar el modelo una sola vez** al iniciar el servicio, nunca en cada petición. Cargarlo tarda varios segundos.
2. **Contar solo `Plier`**, nunca todas las cajas (ver [dos cajas por alicate](#cómo-está-etiquetado-dos-cajas-por-alicate)).
3. **Umbral de confianza** (`conf=0.5`): ajustable. Más bajo detecta más pero con más falsos positivos.
4. **El conteo debe poder corregirse a mano.** El modelo puede equivocarse; en un inventario conviene que el usuario confirme el resultado antes de guardarlo.
5. **Mapear tipos a la base de datos.** Los nombres de clase están en inglés (`Longnose`, `Slip Joint`...). La app debe traducirlos o relacionarlos con sus propios ítems de inventario.
6. **Alcance actual:** el modelo solo reconoce **alicates** de estos 5 tipos. Cualquier otra herramienta no será identificada.

### Alternativas si la app no es Python

- Dejar el servicio de visión como API HTTP (como arriba) y consumirlo desde cualquier lenguaje.
- Exportar el modelo a **ONNX** (`model.export(format="onnx")`) para usarlo en otros entornos.

---

## Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `pip` no se reconoce | Usar `python -m pip install ...`, o reinstalar Python marcando "Add Python to PATH". |
| `unzip: cannot find or open ...` en Colab | Ruta mal escrita. Linux distingue mayúsculas (`Leyca` ≠ `leyca`). Listar con `!ls "/content/drive/MyDrive/"`. |
| `unzip` pregunta `replace ...?` | Ya se había extraído antes. Usar `rm -rf` de las carpetas previas y `unzip -o`. |
| `ls: cannot access '/content/pliers_dataset'` | El zip no trae carpeta contenedora. Extraer con `-d /content/pliers_dataset`. |
| `torch.cuda.is_available()` da `False` | No hay GPU NVIDIA/CUDA local. Entrenar en Colab con T4. |
| `images not found` al entrenar | `path:` del `data.yaml` incorrecto. Debe apuntar a la carpeta que contiene `train/`, `valid/`, `test/`. |
| Error de memoria al entrenar | Bajar `batch` a 8 o 4. |
| `git push` → `Repository not found` | URL o usuario incorrectos, o credenciales de otra cuenta. Ver con `git remote -v`; corregir con `git remote set-url origin <URL>`. |
| `git push` → `rejected (fetch first)` | El remoto tiene commits que no están en local. Ejecutar `git pull --rebase origin main` y luego `git push`. |
| Muchos avisos "Unknown word" en VS Code | Es la extensión Code Spell Checker, no errores de código. Se configura con un `cspell.json`. |
| Primera ejecución muy lenta | Importar `torch` y cargar el modelo toma tiempo; las siguientes son más rápidas. |

---

## Pendientes y siguientes pasos

- [ ] **Probar con fotos reales de los alicates de Leyca** (varios fondos y luces). Es la prueba que decide si el modelo sirve tal cual.
- [ ] Si falla con fotos reales: tomar 50 a 100 fotos propias, etiquetarlas en Roboflow con el mismo esquema de dos cajas y **reentrenar**.
- [ ] Cambiar las **rutas fijas** (`D:/leyca_vision/...`, `C:/Users/...`) de `entrenar.py` por rutas relativas, para que funcione en cualquier equipo.
- [ ] Guardar una copia del `data.yaml` con `path:` relativo en la raíz del repo.
- [ ] Implementar el servicio de integración (`servicio.py`) y definir con el equipo el contrato definitivo del endpoint.
- [ ] Revisar la matriz de confusión de `runs/alicates_v1/` (especialmente `Plier` frente a los demás tipos).
- [ ] Ampliar a otras herramientas e insumos informáticos (nuevo dataset y nuevas clases).
- [ ] Decidir cómo se corrigen manualmente los resultados en la app antes de actualizar el stock.

---

## Flujo de trabajo con Git

```powershell
git pull                      # antes de empezar a trabajar
git add .
git commit -m "Descripción del cambio"
git push
```

No usar `git push --force` salvo que se sepa exactamente qué se va a sobrescribir.

---

## Créditos

- Dataset: **Pliers Computer Vision Model**, por *LBYMF2H Pliers*, vía Roboflow Universe, licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Modelo base: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).
