# Reconhecimento-Facial-Python-TCC  
Este repositório contém o código do meu TCC sobre controle de acesso utilizando reconhecimento facial em subestações elétricas com **Python** e **OpenCV**.

## 📋 Descrição  
O objetivo do projeto é desenvolver um sistema acessível e eficiente para controle de acesso em ambientes industriais, aumentando a segurança e reduzindo erros humanos. Utiliza algoritmos de **visão computacional** para detectar e reconhecer rostos em tempo real, com implementações de **verificação de vivacidade (liveness detection)** para evitar fraudes com fotos ou vídeos.

## 🛠️ Tecnologias Utilizadas  
- **Linguagem:** Python 3  
- **Bibliotecas:**  
  - OpenCV  
  - dlib  
  - face_recognition
  - Pickle 
  - numpy
  - scipy  

## 🚀 Como Executar  
Siga as instruções abaixo para rodar o projeto localmente:

### 1. Clone o repositório  
Se ainda não fez isso, execute no terminal:  
```
git clone https://github.com/seu-usuario/Reconhecimento-Facial-Python-TCC.git
cd Reconhecimento-Facial-Python-TCC
```

### 2. Crie um ambiente virtual (opcional, mas recomendado)
```
python -m venv venv
source venv/bin/activate  # Para Linux/Mac  
venv\Scripts\activate  # Para Windows
```
### 3. Instale as dependências
```
pip install opencv-python dlib face-recognition numpy scipy
```

### 4. Baixe o arquivo shape_predictor_68_face_landmarks.dat
- O arquivo de landmarks faciais é necessário para detectar olhos e pontos da face.
- Coloque o arquivo na pasta do projeto.

## 5. Execute o arquivo principal
Inicie o programa com:

```
python tcc_reconhecimento_facial.py
```
## 📋 Funções do Sistema

- Registro: O sistema solicita seu nome e cargo, captura imagens e salva as informações em um banco de dados (operadores.pkl) dentro de uma pasta chamada (database).

  O usuário passa por três verificações de vivacidade antes que a captura seja iniciada sendo elas:
    - Piscadas: É necessário piscar pelo menos 2 vezes.
    - Direção do olhar: O olhar deve estar focado no centro da câmera.
    - Distância do rosto: O rosto deve estar dentro da faixa de distância correta.

- Login: O sistema faz a verificação biométrica novamente por meio de:
  - Piscadas: É necessário piscar pelo menos 2 vezes.
  - Direção do olhar: O olhar deve estar focado no centro da câmera.
  - Distância do rosto: O rosto deve estar dentro da faixa de distância correta.

Obs: O sistema em questão é apenas um protótipo e é vulnerável a técnicas de fraude avançadas.

## 📦 Requisitos
Python 3.x
Webcam funcional
Sistema com suporte a bibliotecas de visão computacional

🎥 Exemplo de Funcionamento:



![image](https://github.com/user-attachments/assets/2faa11d1-8b20-41e4-af50-291339d2f40b)


## 📞 Contato
Se tiver dúvidas ou sugestões, entre em contato:

E-mail: gabriel.engmack@gmail.com

LinkedIn: https://www.linkedin.com/in/gabriel-dias-silva-perfil/

## 📜 Licença
Este projeto está licenciado sob a MIT License. Consulte o arquivo LICENSE para mais detalhes.
