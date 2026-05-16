from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import requests
from flask_cors import CORS
from collections import Counter

app = Flask(__name__)
CORS(app)

# 🔹 Cargar modelo
model = load_model("sunlit_model_fixed.keras")

# 🔹 Clases con info en español
class_info = {
    'Pepper__bell___Bacterial_spot': {
        "name": "Mancha bacteriana en pimiento",
        "type": "disease",
        "description": "Enfermedad bacteriana que causa manchas oscuras en hojas y frutos.",
        "treatment": [
            "Eliminar hojas infectadas inmediatamente.",
            "Evitar mojar las hojas al regar.",
            "Aplicar bactericidas a base de cobre.",
            "Rotar cultivos para evitar reinfección."
        ]
    },
    'Pepper__bell___healthy': {
        "name": "Pimiento saludable",
        "type": "healthy",
        "description": "La planta se encuentra en buen estado.",
        "treatment": [
            "Mantener riego adecuado.",
            "Revisar periódicamente posibles plagas.",
            "Asegurar buena exposición al sol."
        ]
    },
    'Potato___Early_blight': {
        "name": "Tizón temprano en papa",
        "type": "disease",
        "description": "Hongo que causa manchas marrones con anillos en hojas.",
        "treatment": [
            "Aplicar fungicidas preventivos.",
            "Eliminar hojas afectadas.",
            "Evitar exceso de humedad."
        ]
    },
    'Potato___Late_blight': {
        "name": "Tizón tardío en papa",
        "type": "disease",
        "description": "Enfermedad agresiva que puede destruir la planta rápidamente.",
        "treatment": [
            "Eliminar plantas infectadas completamente.",
            "Aplicar fungicidas sistémicos.",
            "Evitar riego nocturno."
        ]
    },
    'Potato___healthy': {
        "name": "Papa saludable",
        "type": "healthy",
        "description": "La planta está en buen estado.",
        "treatment": [
            "Mantener buen drenaje.",
            "Fertilizar adecuadamente."
        ]
    },
    'Tomato_Bacterial_spot': {
        "name": "Mancha bacteriana en tomate",
        "type": "disease",
        "description": "Provoca manchas en hojas y frutos que afectan la producción.",
        "treatment": [
            "Evitar riego por aspersión.",
            "Aplicar productos a base de cobre.",
            "Eliminar hojas enfermas."
        ]
    },
    'Tomato_Early_blight': {
        "name": "Tizón temprano en tomate",
        "type": "disease",
        "description": "Manchas marrones con patrón en anillos en hojas.",
        "treatment": [
            "Aplicar fungicidas.",
            "Eliminar hojas infectadas.",
            "Mejorar ventilación."
        ]
    },
    'Tomato_Late_blight': {
        "name": "Tizón tardío en tomate",
        "type": "disease",
        "description": "Enfermedad grave que destruye hojas y frutos rápidamente.",
        "treatment": [
            "Eliminar plantas infectadas.",
            "Aplicar fungicidas de acción rápida.",
            "Evitar alta humedad."
        ]
    },
    'Tomato_Leaf_Mold': {
        "name": "Moho en hojas de tomate",
        "type": "disease",
        "description": "Hongo que aparece en ambientes con alta humedad.",
        "treatment": [
            "Reducir humedad ambiental.",
            "Mejorar ventilación.",
            "Aplicar fungicidas."
        ]
    },
    'Tomato_Septoria_leaf_spot': {
        "name": "Mancha foliar por Septoria",
        "type": "disease",
        "description": "Pequeñas manchas oscuras que debilitan la planta.",
        "treatment": [
            "Eliminar hojas afectadas.",
            "Evitar humedad prolongada.",
            "Aplicar fungicidas."
        ]
    },
    'Tomato_Spider_mites_Two_spotted_spider_mite': {
        "name": "Ácaros en tomate",
        "type": "pest",
        "description": "Plaga que succiona la savia y debilita la planta.",
        "treatment": [
            "Aplicar jabón potásico.",
            "Usar insecticidas específicos.",
            "Revisar hojas constantemente."
        ]
    },
    'Tomato__Target_Spot': {
        "name": "Mancha objetivo en tomate",
        "type": "disease",
        "description": "Manchas oscuras bien definidas en hojas.",
        "treatment": [
            "Aplicar fungicidas.",
            "Eliminar hojas dañadas.",
            "Reducir humedad."
        ]
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        "name": "Virus del rizado amarillo",
        "type": "virus",
        "description": "Provoca hojas amarillas y deformadas.",
        "treatment": [
            "Eliminar plantas infectadas.",
            "Controlar mosca blanca.",
            "Usar variedades resistentes."
        ]
    },
    'Tomato__Tomato_mosaic_virus': {
        "name": "Virus del mosaico",
        "type": "virus",
        "description": "Genera patrones irregulares y deformaciones.",
        "treatment": [
            "Eliminar plantas afectadas.",
            "Desinfectar herramientas.",
            "Evitar contacto entre plantas."
        ]
    },
    'Tomato_healthy': {
        "name": "Tomate saludable",
        "type": "healthy",
        "description": "La planta está en buen estado.",
        "treatment": [
            "Mantener riego adecuado.",
            "Fertilizar correctamente."
        ]
    }
}

# 🔹 Preprocesamiento
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# 🔹 Clima
def get_climate(lat, lon):
    try:
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,RH2M,PRECTOT&community=AG&longitude={lon}&latitude={lat}&format=JSON"
        data = requests.get(url, timeout=10).json()
        params = data["properties"]["parameter"]

        return {
            "temperature": list(params["T2M"].values())[-1],
            "humidity": list(params["RH2M"].values())[-1],
            "rain": list(params["PRECTOT"].values())[-1]
        }
    except:
        return {"temperature": None, "humidity": None, "rain": None}

# 🔹 Análisis
def generate_analysis(pred_class, confidence, climate):

    if pred_class == "unknown":
        return {
            "estado": "No identificable",
            "descripcion": "No se pudo identificar claramente la planta.",
            "confianza": round(confidence, 2),
            "recomendaciones": [
                "Tomar una foto más clara",
                "Evitar fondos confusos",
                "Mejorar iluminación"
            ],
            "clima": climate
        }

    info = class_info[pred_class]

    recomendaciones = info["treatment"].copy()

    # 🔥 Inteligencia con clima
    if climate["humidity"] and climate["humidity"] > 80:
        recomendaciones.append("Alta humedad: riesgo elevado de hongos.")

    if climate["temperature"] and climate["temperature"] > 30:
        recomendaciones.append("Temperatura alta: aumentar frecuencia de riego.")

    return {
        "estado": info["name"],
        "tipo": info["type"],
        "descripcion": info["description"],
        "confianza": round(confidence, 2),
        "recomendaciones": recomendaciones,
        "clima": climate
    }

# 🔥 ENDPOINT
@app.route("/predict", methods=["POST"])
def predict():
    try:
        files = request.files.getlist("images")
        lat = request.form.get("lat")
        lon = request.form.get("lon")

        if not files:
            return jsonify({"error": "No se enviaron imágenes"})

        results = []
        predictions_list = []

        climate = get_climate(lat, lon)
        class_keys = list(class_info.keys())

        for file in files:
            try:
                image = Image.open(file).convert("RGB")
                processed = preprocess_image(image)

                prediction = model.predict(processed)
                class_index = np.argmax(prediction)
                confidence = float(np.max(prediction))

                pred_class = class_keys[class_index]

                if confidence < 0.6:
                    pred_class = "unknown"

                predictions_list.append(pred_class)

                results.append(generate_analysis(pred_class, confidence, climate))

            except:
                results.append({"error": "No se pudo procesar la imagen"})

        # 🔥 SUMMARY EN ESPAÑOL
        summary_raw = Counter(predictions_list)
        summary = {}

        for key, value in summary_raw.items():
            if key == "unknown":
                nombre = "No identificable"
            else:
                nombre = class_info[key]["name"]
            summary[nombre] = value

        return jsonify({
            "total_imagenes": len(results),
            "resumen": summary,
            "resultados": results
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/", methods=["GET"])
def home():
    return "API funcionando correctamente"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))