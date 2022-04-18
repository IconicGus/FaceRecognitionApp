# importing libraries

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import numpy as np
import csv
import time
import os

# initializing MTCNN and InceptionResnetV1 

mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40) # keep_all=False
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40) # keep_all=True
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Using webcam recognize face

# loading data.pt file
load_data = torch.load('data2.pt') 
embedding_list = load_data[0] 
name_list = load_data[1] 

#Verificando tamanho da lista de nomes do database
tamanho_lista_name = len(name_list)

#Iniciando lista
qntd_reconhecimentos_individual = []
lista_pessoas_prentes = [] #1 se a pessoa estiver presente e 0 se ela não estiver presente

#Zerando a lista de vezes que cada pessoa foi reconhecida e lista de pessoas presentes
i = 0
while(i < tamanho_lista_name):
    qntd_reconhecimentos_individual.append(0)
    lista_pessoas_prentes.append("faltou")
    i+=1


cam = cv2.VideoCapture(0) 

while True:
    ret, frame = cam.read()
    if not ret:
        print("fail to grab frame, try again")
        break
        
    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True) 
    
    if img_cropped_list is not None:
        boxes, _ = mtcnn.detect(img)
                
        for i, prob in enumerate(prob_list):
            if prob>0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach() 
                
                dist_list = [] # list of matched distances, minimum distance is used to identify the person
                
                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                min_dist = min(dist_list) # get minumum dist value
                min_dist_idx = dist_list.index(min_dist) # get minumum dist index
                name_reconhecido = name_list[min_dist_idx] # get name corrosponding to minimum dist
                
                box = boxes[i] 
                
                original_frame = frame.copy() # storing copy of frame before drawing on it

                #Validação area de face reconhecida
                area_face = (box[2] - box[0])*(box[3] - box[1])
                #print(area_face)

                if(area_face >= 32000):

                    if min_dist<0.99:
                        frame = cv2.putText(frame,'Detectando...Espere', (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)
                        frame = cv2.rectangle(frame, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), (255,0,0), 2)
                        print(name_reconhecido)
                        #Código contador de vezes que cada pessoa foi reconhecida.
                        for name in name_list: 
                            name_list_idx = name_list.index(name) # get index where is the name
                            if name == name_reconhecido:
                                qntd_reconhecimentos_individual[name_list_idx] += 1
                else:
                    frame = cv2.putText(frame, 'Aproxime-se da camera!', (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)

                #Validação se pessoa esta presente ou não

                name_list_idx = name_list.index(name_reconhecido) # get index where is the name most recognized

                print(qntd_reconhecimentos_individual[name_list_idx])

                if(qntd_reconhecimentos_individual[name_list_idx] >= 20):
                    
                    #Contador para passar por todas as pessoas na lista
                    i = 0
                    index_mais_reconhecido = 0 #index da pessoa que mais foi reconhecida

                    frame = cv2.putText(original_frame, name_reconhecido+' esta presente! '+str(min_dist), (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)
                    frame = cv2.rectangle(frame, (int(box[0]),int(box[1])) , (int(box[2]),int(box[3])), (0,0,255), 2)

                    lista_pessoas_prentes[name_list_idx] = "presente"


    cv2.imshow("IMG", frame)
    #print(name)
        
    
    k = cv2.waitKey(1)
    if k%256==27: # ESC
        print('Esc pressed, closing...')

        #Contador para gerar relatório de qntd de vezes que cada nome foi reconhecido
        i = 0
        #print(tamanho_lista_name)
        while(i < tamanho_lista_name):
            #print(str(name_list[i]) + " reconhecido " + str(qntd_reconhecimentos_individual[i]) + " vez(es).")
            i+=1

        #Contador para passar por todas as pessoas na lista
        i = 0
        index_mais_reconhecido = 0 #index da pessoa que mais foi reconhecida

        while(i < tamanho_lista_name):
            
            if(int(qntd_reconhecimentos_individual[i]) >= int(qntd_reconhecimentos_individual[index_mais_reconhecido])): #verifica qual o id da pessoa mais reconhecida dentro da lista
                index_mais_reconhecido = i

            i+=1

        #print(str(name_list[index_mais_reconhecido]) + " foi o mais reconhecido") #printa o nome da pessoa mais reconhecida.
        #print(lista_pessoas_prentes)

        #Código para transformar as listas name_list e lista_pessoas_prentes em uma matriz
        matriz = np.array([[name_list], [lista_pessoas_prentes]])
        m = np.matrix(matriz)
        print(m)

        #código que cria um arquivo CSV da presença
        with open("saida_csv/presenca.csv", "w", newline='') as saida:
            escrever = csv.writer(saida)
            escrever.writerow(name_list)
            escrever.writerow(lista_pessoas_prentes)

        break
        
cam.release()
cv2.destroyAllWindows()