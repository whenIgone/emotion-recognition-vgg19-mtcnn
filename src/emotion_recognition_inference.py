# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from mtcnn import MTCNN


EMOTIONS_TO_USE = ['anger', 'disgust', 'fear', 'happiness', 'sadness']
IMG_SIZE = (224, 224)


def parse_args():
    p = argparse.ArgumentParser(description="Emotion inference with MTCNN + VGG19")
    p.add_argument("--model-path", type=Path, required=True,
                   help="Caminho do .keras (ex: best_model_phase2.keras)")
    p.add_argument("--img", type=Path, default=None,
                   help="Imagem única para inferência")
    p.add_argument("--img-dir", type=Path, default=None,
                   help="Pasta com imagens (.jpg/.jpeg/.png) para inferência em lote")
    return p.parse_args()


detector = MTCNN()


def predict_emotion(image_path: Path, model, emotions):
    """Prediz emoção com detecção MTCNN"""
    try:
        pil_img = Image.open(image_path).convert('RGB')
        img_rgb = np.array(pil_img)
        print(f"✓ Imagem carregada: {img_rgb.shape}")

        faces = detector.detect_faces(img_rgb)

        detected = False
        if len(faces) > 0:
            face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
            x, y, w, h = face['box']

            margin = int(0.15 * max(w, h))
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img_rgb.shape[1] - x, w + 2 * margin)
            h = min(img_rgb.shape[0] - y, h + 2 * margin)

            face_img = img_rgb[y:y + h, x:x + w]
            detected = True
            print(f"✓ Rosto detectado: {w}x{h} px (confiança: {face['confidence']:.2f})")
        else:
            print("⚠️ Nenhum rosto detectado, usando imagem completa")
            face_img = img_rgb

        face_pil = Image.fromarray(face_img).resize(IMG_SIZE)
        img_array = np.array(face_pil) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)
        pred_idx = np.argmax(predictions[0])
        confidence = predictions[0][pred_idx] * 100

        print(f"\n{'='*60}")
        print(f"Predicted: {emotions[pred_idx].upper()}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"{'='*60}\n")

        if detected:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

            rect_img = img_rgb.copy()
            cv2.rectangle(rect_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            ax1.imshow(rect_img); ax1.axis('off')
            ax1.set_title('Original (Rosto Detectado)', fontsize=12)

            ax2.imshow(face_pil); ax2.axis('off')
            ax2.set_title(f'{emotions[pred_idx].upper()} ({confidence:.1f}%)',
                          fontsize=14, fontweight='bold')

            colors = ['#ef553b', '#cccc00', '#63be7b', '#00bfc4', '#ab63fa']
            ax3.barh(emotions, predictions[0], color=colors)
            ax3.set_xlabel('Probability', fontsize=12)
            ax3.set_title('Emotion Probabilities', fontsize=14)
            ax3.set_xlim(0, 1)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.imshow(face_pil); ax1.axis('off')
            ax1.set_title(f'{emotions[pred_idx].upper()} ({confidence:.1f}%)',
                          fontsize=14, fontweight='bold')

            colors = ['#ef553b', '#cccc00', '#63be7b', '#00bfc4', '#ab63fa']
            ax2.barh(emotions, predictions[0], color=colors)
            ax2.set_xlabel('Probability', fontsize=12)
            ax2.set_title('Emotion Probabilities', fontsize=14)
            ax2.set_xlim(0, 1)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Erro: {str(e)}")


def main():
    args = parse_args()
    model = load_model(args.model_path)

    if args.img_dir:
        directory = args.img_dir
        image_paths = (
            list(directory.glob("*.jpg")) +
            list(directory.glob("*.jpeg")) +
            list(directory.glob("*.png"))
        )

        if len(image_paths) == 0:
            print(f"❌ Nenhuma imagem encontrada em: {directory}")
            return

        print(f"✓ {len(image_paths)} imagem(s) encontrada(s)\n")
        for filepath in image_paths:
            print(f"\n{'='*60}")
            print(f"Processando: {filepath.name}")
            print(f"{'='*60}")
            predict_emotion(filepath, model, EMOTIONS_TO_USE)

    if args.img:
        predict_emotion(args.img, model, EMOTIONS_TO_USE)

    if not args.img_dir and not args.img:
        print("⚠️ Use --img ou --img-dir para rodar a inferência.")


if __name__ == "__main__":
    main()
