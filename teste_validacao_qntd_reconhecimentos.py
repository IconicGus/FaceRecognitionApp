# importing libraries

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image

name_reconhecido = "gustavo"

# loading data.pt file
load_data = torch.load(r'C:\Users\gusta\OneDrive\IFSP SJC\Iniciação Científica\OpenCV\MTCNN - learning\FaceRecognitionMTCNN001\data1.pt') 
embedding_list = load_data[0] 
name_list = load_data[1] 

#Verificando tamanho da lista de nomes do database
tamnaho_lista_name = len(name_list)
#print(tamnaho_lista_name)

#Iniciando lista
qntd_reconhecimentos_individual = []

#Zerando a lista de vezes que cada pessoa foi reconhecida
i = 0
while(i < tamnaho_lista_name):
    qntd_reconhecimentos_individual.append(0)
    i+=1

#print(qntd_reconhecimentos_individual)

#Código contador de vezes que cada pessoa foi reconhecida.
for name in name_list: 
    name_list_idx = name_list.index(name) # get index where is the name
    if name == name_reconhecido:
        qntd_reconhecimentos_individual[name_list_idx] += 1 

#Contador para gerar relatório de qntd de vezes que cada nome foi reconhecido
i = 0
while(i < tamnaho_lista_name):
    print(str(name_list[i]) + " reconhecido " + str(qntd_reconhecimentos_individual[i]) + " vez(es).")
    #print(name_list[i])
    i+=1