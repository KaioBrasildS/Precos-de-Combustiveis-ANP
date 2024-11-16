import torch
import numpy as np

# Função kernel (RBF ajustado com a equação 21.39)
def Ridgkernel(x1, x2, sigma_0=1.0, l=1.0):
    diff = np.array(x1, dtype=float) - np.array(x2, dtype=float)
    return sigma_0**2 * np.exp(-np.dot(diff, diff) / (2 * l**2))

# Implementação da Regressão Ridge com SGD para Séries Temporais Online usando PyTorch
class RidgeSGDKernelTorch:
    def __init__(self, eta=0.01, c=1.0, sigma=1.0):
        self.eta = eta  # taxa de aprendizado
        self.c = c  # parâmetro de regularização
        self.sigma = sigma  # parâmetro do kernel
        self.alpha = None  # pesos (serão inicializados depois)
        self.X_train_tensor = None  # histórico de amostras de treino

    def partial_fit(self, x_new, y_new):
        if self.X_train_tensor is None:
            self.X_train_tensor = torch.tensor([x_new], dtype=torch.float32)  # Inicializa o conjunto de treino
            self.alpha = torch.zeros(1, dtype=torch.float32)  # Inicializa os pesos com um único valor
        else:
            self.X_train_tensor = torch.vstack([self.X_train_tensor, torch.tensor(x_new, dtype=torch.float32)])  # Adiciona nova amostra
            self.alpha = torch.cat([self.alpha, torch.zeros(1)])  # Adiciona um peso para a nova amostra

        n_samples = self.X_train_tensor.shape[0]

        # Calcular a previsão
        if n_samples > 1:
            kernels = torch.exp(-torch.norm(self.X_train_tensor[:n_samples - 1] - \
                self.X_train_tensor[n_samples - 1], dim=1) ** 2 / (2 * self.sigma ** 2))
            pred_n = torch.dot(self.alpha[:n_samples - 1], kernels)
        else:
            pred_n = 0

        # Calcular o erro
        error = y_new - pred_n

        # Atualizar os pesos conforme a equação 21.33:
        self.alpha[:n_samples - 1] = (1 - self.eta * self.c) * self.alpha[:n_samples - 1]
        self.alpha[n_samples - 1] = self.eta * error

    def predict(self, x_new):
        # Prever com base nos dados de treino atuais
        x_new_tensor = torch.tensor(x_new, dtype=torch.float32)
        n_samples = self.X_train_tensor.shape[0]

        if n_samples > 1:
            kernels = torch.exp(-torch.norm(self.X_train_tensor[:n_samples - 1] - x_new_tensor, dim=1) ** 2 / (2 * self.sigma ** 2))
            prediction = torch.dot(self.alpha[:n_samples - 1], kernels)
            return prediction.item()
        else:
            return 0.0
