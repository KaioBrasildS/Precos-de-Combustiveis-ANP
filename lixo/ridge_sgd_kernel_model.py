# ridge_sgd_kernel_model.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_kernels

# Implementação da Regressão Ridge com SGD para Séries Temporais Online usando PyTorch
class RidgeSGDKernelTorch:
    def __init__(self, eta=0.01, c=0.01, kernel='rbf', **kernel_params):
        self.eta = eta  # taxa de aprendizado
        self.c = c  # parâmetro de regularização
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.alpha = None  # pesos (serão inicializados depois)
        self.X_train_tensor = None  # histórico de amostras de treino

    def kernel_function(self, x1, x2):
        # Utiliza `pairwise_kernels` do sklearn para calcular o kernel
        return pairwise_kernels(x1.reshape(1, -1), x2.reshape(1, -1), metric=self.kernel, **self.kernel_params)[0, 0]

    def partial_fit(self, x_new_dict, y_new):
        x_new = torch.tensor(list(x_new_dict.values()), dtype=torch.float32)  # Converter dicionário em tensor
        
        if self.X_train_tensor is None:
            self.X_train_tensor = x_new.unsqueeze(0)  # Inicializa o conjunto de treino com o tensor
            self.alpha = torch.zeros(1, dtype=torch.float32)  # Inicializa os pesos com um único valor
        else:
            self.X_train_tensor = torch.vstack([self.X_train_tensor, x_new])  # Adiciona nova amostra
            self.alpha = torch.cat([self.alpha, torch.zeros(1)])  # Adiciona um peso para a nova amostra

        n_samples = self.X_train_tensor.shape[0]

        # Calcular a previsão
        if n_samples > 1:
            kernels = torch.tensor([self.kernel_function(self.X_train_tensor[i].numpy(), x_new.numpy()) for i in range(n_samples - 1)])
            pred_n = torch.dot(self.alpha[:n_samples - 1], kernels)
        else:
            pred_n = 0

        # Calcular o erro
        error = y_new - pred_n

        # Atualizar os pesos conforme a equação 21.33:
        self.alpha[:n_samples - 1] = (1 - self.eta * self.c) * self.alpha[:n_samples - 1]
        self.alpha[n_samples - 1] = self.eta * error

    def predict(self, x_new_dict):
        x_new = torch.tensor(list(x_new_dict.values()), dtype=torch.float32)  # Converter dicionário em tensor
        n_samples = self.X_train_tensor.shape[0]

        if n_samples > 1:
            kernels = torch.tensor([self.kernel_function(self.X_train_tensor[i].numpy(), x_new.numpy()) for i in range(n_samples - 1)])
            prediction = torch.dot(self.alpha[:n_samples - 1], kernels)
            return prediction.item()
        else:
            return 0.0

# Função para executar o modelo
def executar_modelo(X_train_transformed_df, y_train, X_val_transformed_df, y_val, kernel_type='rbf', **kernel_params):
    # Certifique-se de que os dados sejam numéricos
    X_train_transformed_df = X_train_transformed_df.apply(pd.to_numeric, errors='coerce').dropna()
    y_train = pd.to_numeric(y_train, errors='coerce').dropna()

    # Aplicar o StandardScaler aos dados de entrada
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_transformed_df), columns=X_train_transformed_df.columns, index=X_train_transformed_df.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val_transformed_df), columns=X_val_transformed_df.columns, index=X_val_transformed_df.index)

    # Combinar os dados de treino e validação escalados
    X_combined_df = pd.concat([X_train_scaled, X_val_scaled])
    y_combined = pd.concat([y_train, y_val])

    # Converter os dados combinados em uma lista de dicionários
    X_combined_dicts = X_combined_df.to_dict(orient='records')

    # Criar o modelo
    ridge_sgd_torch = RidgeSGDKernelTorch(eta=0.1, c=0.1, kernel=kernel_type, **kernel_params)

    # Treinar o modelo de forma online com os dados combinados em formato de dicionário
    for x_new, y_new in zip(X_combined_dicts, y_combined):
        ridge_sgd_torch.partial_fit(x_new, y_new)

    # Prever com os dados combinados
    y_pred_combined = []
    for x_combined in X_combined_dicts:
        prediction = ridge_sgd_torch.predict(x_combined)
        y_pred_combined.append(prediction)

    # Previsões indexadas
    y_combined_indexed = pd.Series(y_pred_combined, index=y_combined.index)

    # Exibir resultados
    print("Previsões para os dados combinados:", y_pred_combined)
    plt.figure(figsize=(16, 6))
    plt.plot(y_combined, label='Valor Real')
    plt.plot(y_combined_indexed, label='Valor Previsto', linestyle='--')
    plt.title('Comparação entre Valores Reais e Previstos')
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.legend()
    plt.show()
