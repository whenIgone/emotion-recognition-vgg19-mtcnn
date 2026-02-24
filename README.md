Reconhecimento de Emoções Faciais (VGG19) — Treino + Inferência com MTCNN

Projeto de Visão Computacional para classificação de emoções faciais usando Transfer Learning com VGG19 (ImageNet) e treinamento em duas fases.​ Além do treino, inclui um pipeline de inferência que detecta rosto com MTCNN, recorta o maior rosto e então prediz a emoção.
​
Visão geral
* Tarefa: classificação de emoção facial em 5 classes (Anger, Disgust, Fear, Happiness, Sadness).
* Abordagem: backbone VGG19 pré-treinado + “head” denso para classificação.​
* Inferência: detecção de faces com MTCNN e seleção do maior rosto (maior área do bounding box).
​
Dataset
* Fonte: Kaggle — Balanced CK+ Dataset (imagens 75×75 RGB).
* Classes (5): Anger, Disgust, Fear, Happiness, Sadness.
* Split em diretórios separados: train/, val/, test/.

Estrutura esperada (paths atualizados)
Ajuste conforme as modificações de path que fizemos no projeto (padrão abaixo para rodar “portável” a partir da raiz do repositório).

.
├─ data/
│  └─ ck_balanced/
│     ├─ train/
│     │  ├─ Anger/
│     │  ├─ Disgust/
│     │  ├─ Fear/
│     │  ├─ Happiness/
│     │  └─ Sadness/
│     ├─ val/
│     │  └─ (mesmas 5 classes)
│     └─ test/
│        └─ (mesmas 5 classes)
├─ models/
├─ images/
└─ src/
   ├─ train_emotion_vgg19.py
   └─ inference_emotion_mtcnn.py

Modelo:
Backbone (VGG19)
* VGG19 com pesos do ImageNet e include_top=False para usar a rede como extratora de features.​
* Entrada usada no projeto: 224×224×3 (o pipeline faz resize das imagens para esse tamanho).​

Cabeça de classificação (head)
Arquitetura do head (conforme o projeto):
* GlobalAveragePooling2D
* Dropout(0.5)
* Dense(256, ReLU)
* Dropout(0.4)
* Dense(5, Softmax)

Pré-processamento (VGG)
Use preprocess_input da família VGG para preparar as imagens conforme o padrão do ImageNet; ele converte RGB→BGR e centraliza por canal (sem “scaling” adicional).

Treinamento (2 fases):

Fase 1 — “feature extractor”
* Backbone congelado.
* 20 épocas.
* Adam (lr=1e-3).

Fase 2 — fine-tuning
* Descongela as últimas 8 camadas do backbone para refinar no dataset.
* 20 épocas.
* Adam (lr=5e-5).

Callbacks
* EarlyStopping
* ModelCheckpoint (monitorando melhor val_accuracy)
* ReduceLROnPlateau

Avaliação e interpretabilidade (saídas em images/)
O projeto gera prints/figuras em images/ contendo:
* Distribuição por classe em train/val/test.
* Curvas de loss/accuracy com marca do início do fine-tuning.
* Matrizes de confusão (val e test).
* PCA 3D de embeddings extraídos da camada fc1.

Como executar
1) Instalar dependências
Exemplo (ajuste para sua stack/versão):

pip install -r requirements.txt
2) Conferir paths (modificações aplicadas)
Os scripts foram ajustados para trabalhar com paths relativos à raiz do projeto (ex.: data/, models/, images/) em vez de paths locais hardcoded do Windows (ex.: C:\...).

Se você ainda tiver algum path hardcoded, o próximo passo recomendado é parametrizar via CLI (argparse) e/ou variáveis de ambiente.

3) Rodar
Treino:
python src/train_emotion_vgg19.py

Inferência:
python src/inference_emotion_mtcnn.py

Próximos passos
* Parametrizar paths por CLI (argparse) e remover qualquer resquício de paths hardcoded.
* Exportar uma demo (Streamlit/FastAPI).
* Testar generalização em imagens “in the wild”.
