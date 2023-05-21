import numpy as np
import cv2 as cv
import mediapipe as mp
import pandas as pd


def qualletra(idatual, pontos):
    if idatual[0] == 12  and idatual[1] == 8  and idatual[2] ==  11:
        return 'R'
    elif idatual[-1] in [12,8,11,16] and idatual[-2] in [12,8,11,16] and idatual[-3] in [12,8,11,16]:
        if pontos[20][1]<= pontos[19][1] :
            return 'M'
    elif idatual[0] == 10  and idatual[1] == 6  and idatual[2] == 14:
        return 'S'
    elif idatual[0] == 4  and idatual[1] == 10  and idatual[2] == 6  and idatual[3] == 14:
        return 'A'

    elif idatual[0] == 15 and idatual[1] == 11 and idatual[2] == 14:
        return 'C'
    elif idatual[0] == 10  and idatual[1] == 14  and idatual[2] == 6:
        return 'O'
    elif idatual[0] == 5  and idatual[1] == 9  and idatual[2] == 6:
        return '-'


    





# aqui estamos abrindo a camera
cap = cv.VideoCapture(0)
# aqui começamos a utilizar o mp veja que estamos instanciando 
# estamos dizendo que vamos usar a solução para escanear mãos
hand =mp.solutions.hands
# aqui dizemos a quantidade de mãos que queremos escanear
Hand = hand.Hands(max_num_hands=3)
#responsavel por desenhar a mão
mpDraw =mp.solutions.drawing_utils
contador = ''
letra_a_esq = [4, 6, 10, 14, 18, 9, 13, 5, 17, 3, 7, 19, 11, 15, 20, 8, 12, 2, 16, 1, 0]
letra_a_dir = [10, 14, 18, 6, 17, 13, 9, 4, 5, 19, 15, 11, 3, 7, 20, 16, 12, 8, 2, 0, 1]

while True:
    #pega a imagem da camera
    check,img = cap.read()
    #recebemos a imagem camera como brg e temos que mudar para rgb para poder começar a indentificar e desenhar as mãos
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    #aqui temos o processamento dessa nova imagem
    results =Hand.process(imgRGB)
    #extraindo os pontos que queremos verificar dentro do desenho da mão
    handsPoints = results.multi_hand_landmarks
    # precisamos da dimanesão da imagem em pixels
    altura, largura, _ = img.shape
    #vamos usar um array de nome pontos ele vai ser usada logo 
    pontos = []
    lista_cy = []
    lista_id = []
    ordem_dos_pontos = []
    # nos primeiros frames não tem mao na tela enão ele não conseguiterar nada logo gera um erro e por isso samos  if 
    if handsPoints:
        #aqui estamos pegando a coordenada de cada ponto da mão vai 
        for points in handsPoints:
            #printa os landmarks
            # print(points)
            # agora estamos mostrando os desenhos da mão
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)
            #agora temos noção dos pontos e podemos gerar uma lógica em cima disso
            # o id representa o x e cord o y 
            for id,cord in enumerate(points.landmark):
                lista_id.append(id)
                #estamos transformando as cordenadas em pixels para facilitar o trabalho
                cx,cy = int(cord.x*largura), int(cord.y*altura)
                # vai mostrar todos os pontos da mão de forma enumerada
                cv.putText(img, str(id), (cx, cy+10), cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2) 
                #nos passa a cordenada de cada ponto
                pontos.append((cx, cy))
                lista_cy.append(cy)
                ordem_dos_pontos= {'id': lista_id,'cy':lista_cy}
    if pontos:
        #m
        df = pd.DataFrame(ordem_dos_pontos)
        df = df.sort_values('cy')
        idatual = list(df['id'])
        print(idatual)
        nl = ''
        nl =qualletra(idatual, pontos)
        print(nl)
        if nl != None:
            if contador == '':
                contador = nl
            elif contador != nl  and len(contador) == 1 :
                contador += nl
            elif len(contador) > 1 and contador[-1] != nl:
                contador += nl
        if nl == '-':
            contador = ''
        
    


    cv.rectangle(img,(80,10),(600,120),(255,255,255),-1)
    cv.putText(img,str(contador),(100, 100), cv.FONT_HERSHEY_SIMPLEX,4,(255, 0,0),5)
    #aqui estamos abrindo a imagem 
    #mostra a tela com o nome de imagem 
    cv.imshow("IA de Hand", img)
    #aqui é o tempo de delay 
    cv.waitKey(1)
    







