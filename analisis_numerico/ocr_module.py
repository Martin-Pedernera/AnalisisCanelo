"""
ocr_module.py — Procesamiento de imágenes con Tesseract + OpenCV
"""
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
import re
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocesa la imagen para mejorar la detección OCR:
    - Convierte a escala de grises
    - Aplica umbralización adaptativa
    - Elimina ruido
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Redimensionar si es muy pequeña (mejora OCR)
    h, w = gray.shape
    if w < 800:
        scale = 800 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Reducción de ruido
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Umbralización adaptativa (mejor para iluminación variable)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    # Morphological opening para limpiar puntos
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return cleaned


def extract_text_from_image(image_bytes: bytes) -> dict:
    """
    Extrae texto de una imagen usando Tesseract OCR.
    Devuelve el texto crudo y el texto normalizado.
    """
    try:
        processed = preprocess_image(image_bytes)
        pil_img = Image.fromarray(processed)

        # Configuración de Tesseract: modo página = texto disperso, idioma español + inglés
        custom_config = r'--oem 3 --psm 6 -l eng+spa'
        raw_text = pytesseract.image_to_string(pil_img, config=custom_config)

        # También intentar con imagen sin procesar por si el procesado empeora
        pil_original = Image.open(io.BytesIO(image_bytes)).convert('L')
        raw_text_alt = pytesseract.image_to_string(pil_original, config=custom_config)

        # Quedarse con el que tiene más contenido útil
        text = raw_text if len(raw_text.strip()) >= len(raw_text_alt.strip()) else raw_text_alt

        normalized = normalize_math_text(text)

        return {
            "success": True,
            "raw_text": text.strip(),
            "normalized_text": normalized,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "raw_text": "",
            "normalized_text": "",
            "error": str(e)
        }


def normalize_math_text(text: str) -> str:
    """
    Normaliza el texto extraído por OCR para expresiones matemáticas comunes.
    Corrige errores típicos de OCR en notación matemática.
    """
    # Limpiar saltos y espacios excesivos
    text = re.sub(r'\s+', ' ', text).strip()

    # Correcciones comunes de OCR en notación matemática
    replacements = {
        # Letras confundidas
        'Oo': '0', '|': '1',
        # Operadores
        '×': '*', '÷': '/',
        # Potencias escritas como texto
        r'\^2': '**2', r'\^3': '**3',
        # Fracciones comunes mal leídas
        '½': '1/2', '¼': '1/4', '¾': '3/4',
        # Símbolos matemáticos
        'π': 'pi', 'Σ': 'sum', '∫': 'integral',
        '√': 'sqrt', '∞': 'inf',
        # Espacios alrededor de operadores
        r' \* ': '*', r' / ': '/',
    }

    for old, new in replacements.items():
        if old.startswith(r'\ ') or old.startswith(r'\^'):
            text = re.sub(old, new, text)
        else:
            text = text.replace(old, new)

    return text


def image_bytes_from_uploaded(uploaded_file) -> bytes:
    """Convierte el archivo subido por Streamlit a bytes."""
    return uploaded_file.getvalue()
