import torch
import numpy as np
import pandas as pd

class RidgeSGDKernelTorchDict:
    def __init__(self, eta=0.01, c=1.0, sigma=1.0):
        self.eta = eta  # Taxa de aprendizado
        self.c = c  # Parâmetro de regularização
        self.sigma = sigma  # Parâmetro do kernel
        self.alpha_dict = {}  # Dicionário para armazenar os pesos
        self.X_train_dict = {}  # Dicionário para armazenar os dados de treino

    def partial_fit(self, x_new, y_new, key):
        """
        Adiciona uma nova entrada ao modelo e atualiza os pesos.

        Args:
        - x_new: nova amostra (array ou lista de características).
        - y_new: valor alvo associado à amostra.
        - key: identificador único para a amostra.
        """
        x_new_tensor = torch.tensor(x_new, dtype=torch.float32)
        self.X_train_dict[key] = x_new_tensor  # Adiciona nova entrada ao conjunto de treino
        if key not in self.alpha_dict:
            self.alpha_dict[key] = torch.tensor(0.0, dtype=torch.float32)  # Inicializa o peso correspondente

        # Calcular a previsão (pred_n)
        pred_n = torch.tensor(0.0, dtype=torch.float32)  # Inicializa como tensor PyTorch
        for k, x_train in self.X_train_dict.items():
            if k != key:  # Considera apenas os pontos anteriores
                kernel_value = torch.exp(-torch.norm(x_train - x_new_tensor) ** 2 / (2 * self.sigma ** 2))
                pred_n += self.alpha_dict[k] * kernel_value

        # Calcular o erro
        error = torch.tensor(y_new, dtype=torch.float32) - pred_n

        # Atualizar os pesos
        for k in list(self.alpha_dict.keys()):
            if k != key:
                self.alpha_dict[k] *= (1 - self.eta * self.c)  # Atualização regularizada
        self.alpha_dict[key] = self.eta * error  # Atualiza o peso para a nova amostra

        # Retorna a previsão (pred_n) para essa nova amostra
        return pred_n.item()  # pred_n é garantido como tensor PyTorch

