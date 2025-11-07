# 📘 Descrição Geral

Este projeto tem como objetivo descobrir a função misteriosa criada pelo professor utilizando técnicas de Inteligência Artificial embarcadas em um ESP32.
O processo envolve a coleta de dados via requisição HTTP, o tratamento de ruídos e outliers, e o treinamento de um modelo de aprendizado de máquina capaz de inferir o comportamento da função oculta.

O projeto faz parte de uma série de experimentos práticos de IA embarcada, explorando o fluxo completo de:

coleta → pré-processamento → modelagem → inferência → implantação em hardware (ESP32)

🧩 Etapas do Projeto
1. Coleta de Dados

Os dados foram obtidos via requisição HTTP a um endpoint fornecido pelo professor.\
Cada requisição retornava pares de valores de entrada e saída de uma função misteriosa.\
Os dados continham ruídos e outliers, simulando medições reais com imperfeições.

2. Tratamento dos Dados

Aplicamos limpeza e normalização para remover ruídos e suavizar outliers.\
Foram implementados filtros e transformações para preparar os dados para o modelo.

3. Treinamento do Modelo

Os dados tratados foram utilizados para treinar um modelo de aprendizado de máquina.\
O modelo foi projetado para aproximar a função misteriosa com base em exemplos observados.\
Foram testados diferentes modelos e ajustes de hiperparâmetros.

4. Implantação no ESP32

Após o treinamento, o modelo foi convertido e embarcado no ESP32, permitindo inferências locais.\
O dispositivo passou a receber novas entradas e prever a saída estimada da função diretamente no microcontrolador.\
O objetivo foi demonstrar a viabilidade da IA embarcada, mesmo com recursos computacionais limitados.

🧠 Tecnologias Utilizadas
 - ESP32 DevKit
 - Linguagem C / C++
 - Treinamento do modelo Python (Google Colab / TensorFlow / Scikit-learn)
 - Comunicação	HTTP Requests
 - Pré-processamento	NumPy, Pandas, Matplotlib
 - Inferência embarcada  com	TensorFlow Lite Micro
