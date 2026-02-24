# Reconhecimento de Emoções Faciais (VGG19) — Treino + Inferência com MTCNN

Projeto de Visão Computacional para **classificação de emoções faciais** usando Transfer Learning com **VGG19 (ImageNet)** e treinamento em duas fases.  
Além do treino, inclui um pipeline de **inferência** que detecta rosto com **MTCNN**, recorta o maior rosto e então prediz a emoção.

## Dataset
- Kaggle: Balanced CK+ Dataset (75×75 RGB)
- Classes (5): Anger, Disgust, Fear, Happiness, Sadness
- Split: `train/val/test` (diretórios separados)

## Modelo
- Backbone: VGG19 (ImageNet, `include_top=False`)
- Head: GlobalAveragePooling2D → Dropout(0.5) → Dense(256, ReLU) → Dropout(0.4) → Dense(5, Softmax)

## Treinamento (2 fases)
- Fase 1: backbone congelado, 20 épocas, Adam (lr=1e-3)
- Fase 2: fine-tuning das últimas 8 camadas, 20 épocas, Adam (lr=5e-5)
- Callbacks: EarlyStopping, ModelCheckpoint (melhor val_accuracy), ReduceLROnPlateau

## Avaliação e interpretabilidade (prints em /images)
- Distribuição por classe em train/val/test
- Curvas de loss/accuracy com marco do início do fine-tuning
- Matrizes de confusão (val e test)
- PCA 3D de embeddings extraídos da camada `fc1`

## Inferência com detecção de rosto (MTCNN)
O script de inferência:
- Detecta rostos com MTCNN
- Seleciona o maior rosto e aplica margem (~15%) no recorte
- Faz resize para 224×224, normaliza e prediz emoção
- Exibe: bounding box + face recortada + gráfico de probabilidades

## Como executar
1. Instale dependências
2. Ajuste os paths (estão como paths locais no Windows)
3. Rode:
- Treino: `python src/train_emotion_vgg19.py`
- Inferência: `python src/inference_emotion_mtcnn.py`

## Próximos passos
- Parametrizar paths por CLI (argparse) e remover paths hardcoded
- Exportar uma demo (Streamlit/FastAPI)
- Testar generalização em imagens “in the wild”
