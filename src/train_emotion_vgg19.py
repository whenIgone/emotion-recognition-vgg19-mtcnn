# -*- coding: utf-8 -*-
"""
Reconhecimento de Emoções Faciais com VGG19 (CK+) — Treino + Avaliação + PCA 3D

Original notebook:
https://colab.research.google.com/drive/1SiyKHaogCInbxTrcNanGevhhuJcw_jTa

Dataset (Kaggle):
https://www.kaggle.com/datasets/dollyprajapati182/balanced-ck-dataset-7575-rgb

Uso (exemplos):
  python src/train_emotion_vgg19.py --data-dir "C:/caminho/archive" --out-dir results
"""

from __future__ import annotations

from pathlib import Path
import argparse
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# =========================
# Configurações (iguais ao original)
# =========================
SEED = 42
EMOTIONS_TO_USE = ["Anger", "Disgust", "Fear", "Happiness", "Sadness"]
EMOTIONS_TO_REMOVE = ["contempt", "surprise"]  # apenas informativo (original)
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 20

LR_PHASE1 = 1e-3
LR_PHASE2 = 5e-5

FINE_TUNE_LAST_N_LAYERS = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VGG19 emotion recognition on CK+ (balanced)")
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Pasta raiz do dataset (deve conter train/ val/ test/ com subpastas por classe)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Pasta para salvar modelos, figuras e métricas",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Se definido, não chama plt.show() (útil em execução headless).",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.config.experimental.enable_op_determinism()


def enable_gpu_memory_growth() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Memory growth habilitado para {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(e)


def count_images_per_emotion(split_dir: Path, emotions: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for emotion in emotions:
        emotion_dir = split_dir / emotion
        if emotion_dir.exists():
            # conta arquivos em qualquer extensão
            counts[emotion] = len([p for p in emotion_dir.iterdir() if p.is_file()])
        else:
            counts[emotion] = 0
    return counts


def save_distribution_plot(
    train_counts: dict[str, int],
    val_counts: dict[str, int],
    test_counts: dict[str, int],
    out_path: Path,
    show: bool,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (split_name, counts) in enumerate([("Train", train_counts), ("Val", val_counts), ("Test", test_counts)]):
        ax = axes[idx]
        emotions = list(counts.keys())
        values = list(counts.values())

        bars = ax.bar(emotions, values, color=sns.color_palette("husl", len(emotions)))
        ax.set_title(f"{split_name} Set Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Emotion", fontsize=12)
        ax.set_ylabel("Number of Images", fontsize=12)
        ax.tick_params(axis="x", rotation=45)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def combine_histories(hist1, hist2) -> dict[str, list[float]]:
    combined: dict[str, list[float]] = {}
    for key in hist1.history.keys():
        combined[key] = hist1.history[key] + hist2.history[key]
    return combined


def save_training_curves(combined_history: dict[str, list[float]], fine_tune_epoch: int, out_path: Path, show: bool):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(combined_history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(combined_history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].axvline(x=fine_tune_epoch, color="red", linestyle="--", label="Início Fine-tuning", linewidth=2)
    axes[0].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].legend(loc="upper right", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(combined_history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[1].plot(combined_history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[1].axvline(x=fine_tune_epoch, color="red", linestyle="--", label="Início Fine-tuning", linewidth=2)
    axes[1].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def save_confusion_matrices(
    y_val_true: np.ndarray,
    y_val_pred: np.ndarray,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    labels: list[str],
    out_path: Path,
    show: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    cm_val = confusion_matrix(y_val_true, y_val_pred)
    disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=labels)
    disp_val.plot(ax=axes[0], cmap="Blues", values_format="d")
    axes[0].set_title("Validation Set Confusion Matrix", fontsize=14, fontweight="bold")
    axes[0].grid(False)

    cm_test = confusion_matrix(y_test_true, y_test_pred)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=labels)
    disp_test.plot(ax=axes[1], cmap="Greens", values_format="d")
    axes[1].set_title("Test Set Confusion Matrix", fontsize=14, fontweight="bold")
    axes[1].grid(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def save_embeddings_pca_3d_plotly(
    embeddings: np.ndarray,
    y_test: np.ndarray,
    class_labels: list[str],
    out_html: Path,
) -> None:
    # Mantém a lógica do original: PCA 3 componentes e Plotly 3D
    import plotly.graph_objects as go

    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(embeddings)
    variance = pca.explained_variance_ratio_

    colors = {
        "Anger": "rgb(239, 85, 59)",
        "Disgust": "rgb(204, 204, 0)",
        "Fear": "rgb(99, 190, 123)",
        "Happiness": "rgb(0, 191, 196)",
        "Sadness": "rgb(171, 99, 250)",
    }

    fig = go.Figure()

    for idx, emotion in enumerate(class_labels):
        mask = y_test == idx
        fig.add_trace(
            go.Scatter3d(
                x=embeddings_3d[mask, 0],
                y=embeddings_3d[mask, 1],
                z=embeddings_3d[mask, 2],
                mode="markers",
                name=emotion,
                marker=dict(
                    size=5,
                    color=colors.get(emotion, "gray"),
                    opacity=0.7,
                    line=dict(width=0.5, color="white"),
                ),
                hovertemplate=f"<b>{emotion}</b><br>"
                + "PC1: %{x:.2f}<br>"
                + "PC2: %{y:.2f}<br>"
                + "PC3: %{z:.2f}<br>"
                + "<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="3D Visualization of Emotion Embeddings (PCA)<br>"
            + f"<sub>Variância explicada: {variance.sum()*100:.2f}%</sub>",
            x=0.5,
            xanchor="center",
        ),
        scene=dict(
            xaxis_title=f"PC1 ({variance[0]*100:.2f}%)",
            yaxis_title=f"PC2 ({variance[1]*100:.2f}%)",
            zaxis_title=f"PC3 ({variance[2]*100:.2f}%)",
            bgcolor="rgba(240, 240, 245, 0.9)",
            xaxis=dict(backgroundcolor="rgb(230, 230,230)"),
            yaxis=dict(backgroundcolor="rgb(230, 230,230)"),
            zaxis=dict(backgroundcolor="rgb(230, 230,230)"),
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=60),
    )

    # salva HTML interativo (melhor para repo do que só fig.show())
    fig.write_html(str(out_html), include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()

    # Visual configs (como no original)
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    # Seeds / determinismo
    set_seed(SEED)
    enable_gpu_memory_growth()

    print(f"✓ TensorFlow versão: {tf.__version__}")
    print(f"✓ GPU disponível: {tf.config.list_physical_devices('GPU')}")

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    print(f"✓ Train directory: {train_dir.exists()}")
    print(f"✓ Val directory: {val_dir.exists()}")
    print(f"✓ Test directory: {test_dir.exists()}")

    if not (train_dir.exists() and val_dir.exists() and test_dir.exists()):
        raise FileNotFoundError("Estrutura esperada: data-dir/{train,val,test}/<Emotion>/...")

    print(f"✓ Emoções selecionadas: {EMOTIONS_TO_USE}")
    print(f"✗ Emoções removidas: {EMOTIONS_TO_REMOVE}")

    # Distribuição
    train_counts = count_images_per_emotion(train_dir, EMOTIONS_TO_USE)
    val_counts = count_images_per_emotion(val_dir, EMOTIONS_TO_USE)
    test_counts = count_images_per_emotion(test_dir, EMOTIONS_TO_USE)

    df_counts = pd.DataFrame({"Train": train_counts, "Val": val_counts, "Test": test_counts})
    df_counts["Total"] = df_counts.sum(axis=1)
    df_counts.loc["TOTAL"] = df_counts.sum()

    print("\n" + "=" * 60)
    print("📊 DISTRIBUIÇÃO DE IMAGENS POR EMOÇÃO")
    print("=" * 60)
    print(df_counts)
    print("=" * 60)
    print(f"\n✓ Total de imagens: {int(df_counts.loc['TOTAL', 'Total'])}")

    # Plot distribuição
    dist_path = out_dir / "data_distribution.png"
    save_distribution_plot(train_counts, val_counts, test_counts, dist_path, show=(not args.no_plots))
    print("\n✓ Gráfico de distribuição salvo em:", dist_path)

    # Verificação de balanceamento (igual ao original)
    print("\n" + "=" * 60)
    print("⚖️  VERIFICAÇÃO DE BALANCEAMENTO")
    print("=" * 60)
    for split_name, counts in [("Train", train_counts), ("Val", val_counts), ("Test", test_counts)]:
        total = sum(counts.values())
        print(f"\n{split_name} Set:")
        for emotion, count in counts.items():
            percentage = (count / total) * 100 if total > 0 else 0.0
            print(f"  {emotion:12s}: {count:4d} imagens ({percentage:5.2f}%)")
        percentages = [(count / total) * 100 for count in counts.values()] if total > 0 else [0.0] * len(counts)
        std_dev = float(np.std(percentages))
        print(f"  {'Desvio padrão':12s}: {std_dev:.3f}% (quanto menor, mais balanceado)")

    print("\n" + "=" * 60)
    print("✓ Dataset está balanceado - NÃO será necessário usar class_weight")
    print("=" * 60)

    # Geradores
    print(f"\n📐 Configuração:")
    print(f"  - Tamanho da imagem: {IMG_SIZE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Número de classes: {len(EMOTIONS_TO_USE)}")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # flow_from_directory aceita str path; passamos str(...) para evitar inconsistências
    train_generator = train_datagen.flow_from_directory(
        directory=str(train_dir),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=EMOTIONS_TO_USE,
        shuffle=True,
        seed=SEED,
    )
    val_generator = val_test_datagen.flow_from_directory(
        directory=str(val_dir),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=EMOTIONS_TO_USE,
        shuffle=False,
        seed=SEED,
    )
    test_generator = val_test_datagen.flow_from_directory(
        directory=str(test_dir),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=EMOTIONS_TO_USE,
        shuffle=False,
        seed=SEED,
    )

    print(f"\n✓ Geradores criados:")
    print(f"  - Train samples: {train_generator.samples}")
    print(f"  - Val samples: {val_generator.samples}")
    print(f"  - Test samples: {test_generator.samples}")
    print(f"\n✓ Mapeamento de classes: {train_generator.class_indices}")

    # Modelo base
    base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    print("✓ VGG19 base carregada do ImageNet")
    print(f"  - Total de camadas: {len(base_model.layers)}")
    print(f"  - Shape de saída: {base_model.output_shape}")

    # Head (igual ao original)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu", name="fc1")(x)
    x = Dropout(0.4)(x)
    predictions = Dense(len(EMOTIONS_TO_USE), activation="softmax", name="predictions")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    print(f"\n✓ Modelo completo criado:")
    print(f"  - Total de camadas: {len(model.layers)}")
    print(f"  - Shape de entrada: {model.input_shape}")
    print(f"  - Shape de saída: {model.output_shape}")
    model.summary()

    # ===== Fase 1 =====
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=LR_PHASE1), loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks_phase1 = [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            filepath=str(out_dir / "best_model_phase1.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
    ]

    print("\n" + "=" * 60)
    print("🚀 INICIANDO FASE 1: BACKBONE CONGELADO")
    print("=" * 60)
    print(f"⏱️  Épocas: {EPOCHS_PHASE1}")
    print("📊 Sem class_weight (dataset balanceado)\n")

    history_phase1 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS_PHASE1,
        callbacks=callbacks_phase1,
        verbose=1,
    )

    print("\n" + "=" * 60)
    print("✓ FASE 1 CONCLUÍDA")
    print("=" * 60)

    # ===== Fase 2 =====
    for layer in base_model.layers[:-FINE_TUNE_LAST_N_LAYERS]:
        layer.trainable = False
    for layer in base_model.layers[-FINE_TUNE_LAST_N_LAYERS:]:
        layer.trainable = True

    print(f"\n✓ Últimas {FINE_TUNE_LAST_N_LAYERS} camadas da VGG19 descongeladas")
    print("\n📋 Camadas descongeladas:")
    for layer in base_model.layers[-FINE_TUNE_LAST_N_LAYERS:]:
        print(f"  - {layer.name}")

    model.compile(optimizer=Adam(learning_rate=LR_PHASE2), loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks_phase2 = [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            filepath=str(out_dir / "best_model_phase2.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
    ]

    print("\n" + "=" * 60)
    print("🔥 INICIANDO FASE 2: FINE-TUNING")
    print("=" * 60)
    print(f"⏱️  Épocas: {EPOCHS_PHASE2}")
    print("📊 Sem class_weight (dataset balanceado)\n")

    history_phase2 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS_PHASE2,
        callbacks=callbacks_phase2,
        verbose=1,
    )

    print("\n" + "=" * 60)
    print("✓ FASE 2 CONCLUÍDA")
    print("=" * 60)

    # Curvas (igual ao original)
    combined_history = combine_histories(history_phase1, history_phase2)
    curves_path = out_dir / "training_curves.png"
    save_training_curves(combined_history, fine_tune_epoch=EPOCHS_PHASE1 - 1, out_path=curves_path, show=(not args.no_plots))
    print(f"\n✓ Curvas de treinamento salvas em: {curves_path}")

    # Avaliação teste (igual ao original)
    print("\n" + "=" * 60)
    print("🎯 AVALIAÇÃO NO CONJUNTO DE TESTE")
    print("=" * 60 + "\n")

    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

    print(f"\n{'='*60}")
    print("📊 RESULTADO FINAL")
    print(f"{'='*60}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"{'='*60}")

    # Predições teste
    test_generator.reset()
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes

    # Classification report
    report = classification_report(y_true, y_pred, target_names=EMOTIONS_TO_USE, digits=4)
    print("\n" + "=" * 60)
    print("📋 CLASSIFICATION REPORT")
    print("=" * 60 + "\n")
    print(report)

    # Salvar métricas (igual ao original)
    results_path = out_dir / "test_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"\n✓ Resultados salvos em: {results_path}")

    # Matrizes de confusão (val e test)
    val_generator.reset()
    y_val_pred_probs = model.predict(val_generator, verbose=1)
    y_val_pred = np.argmax(y_val_pred_probs, axis=1)
    y_val_true = val_generator.classes

    conf_path = out_dir / "confusion_matrices.png"
    save_confusion_matrices(
        y_val_true=y_val_true,
        y_val_pred=y_val_pred,
        y_test_true=y_true,
        y_test_pred=y_pred,
        labels=EMOTIONS_TO_USE,
        out_path=conf_path,
        show=(not args.no_plots),
    )
    print(f"\n✓ Matrizes de confusão salvas em: {conf_path}")

    # Embeddings + PCA 3D (corrigido para rodar em .py)
    embedding_model = Model(inputs=model.input, outputs=model.get_layer("fc1").output)
    print("\n✓ Modelo de embeddings criado")
    print(f"  - Shape de saída: {embedding_model.output_shape}")

    test_generator.reset()
    embeddings = embedding_model.predict(test_generator, verbose=1)
    print(f"\n✓ Embeddings extraídos: {embeddings.shape}")

    # Define o que o notebook "assumia"
    y_test = y_true
    class_labels = EMOTIONS_TO_USE

    pca_html_path = out_dir / "embeddings_pca_3d.html"
    save_embeddings_pca_3d_plotly(embeddings, y_test, class_labels, pca_html_path)
    print(f"✓ PCA 3D (Plotly) salvo em: {pca_html_path}")

    # Salvar modelo final (igual ao original, mas com Path)
    final_model_path = out_dir / "emotion_recognition_vgg19_final.keras"
    model.save(final_model_path)
    print(f"\n✓ Modelo final salvo em: {final_model_path}")

    # Listar arquivos salvos (igual ao original)
    print("\n" + "=" * 60)
    print("📁 ARQUIVOS SALVOS")
    print("=" * 60)
    for fp in sorted(out_dir.iterdir()):
        if fp.is_file():
            size_mb = fp.stat().st_size / (1024 * 1024)
            print(f"  {fp.name:40s} ({size_mb:.2f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
