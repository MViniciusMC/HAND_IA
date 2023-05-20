import numpy as np
import cv2 as cv
import mediapipe as mp



# aqui estamos abrindo a camera
cap = cv.VideoCapture(0)
# aqui começamos a utilizar o mp veja que estamos instanciando 
# estamos dizendo que vamos usar a solução para escanear mãos
hand =mp.solutions.hands
# aqui dizemos a quantidade de mãos que queremos escanear
Hand = hand.Hands(max_num_hands=1)
#responsavel por desenhar a mão
mpDraw =mp.solutions.drawing_utils
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
                #estamos transformando as cordenadas em pixels para facilitar o trabalho
                cx,cy = int(cord.x*largura), int(cord.y*altura)
                # vai mostrar todos os pontos da mão de forma enumerada
                # cv.putText(img, str(id), (cx, cy+10), cv.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),2) 
                #nos passa a cordenada de cada ponto
                pontos.append((cx, cy))
                #assim podemos gerar uma logica onde o ponto superior do dedo estiver a baixo do ponto mais baixo o dedo esta abaixado
    #todos os dedos menos o dedão a logica para ele é diferente 
    dedos = [8, 12, 16, 20]
    contador = 0
    if pontos:
        # estamos corendo para cada dedo da mão e pontos[x][1] diz para pegar o valor altura
        #para o dedão
        if pontos[4][0] < pontos[2][0]:
            contador +=1
        #para os demais dedos
        for dedo in dedos:
            if pontos[dedo][1] < pontos[dedo-2][1]:
                contador+=1
    cv.rectangle(img,(80,10),(200,100),(255,255,255),-1)
    cv.putText(img,str(contador),(100, 100), cv.FONT_HERSHEY_SIMPLEX,4,(255, 0,0),5)
    #aqui estamos abrindo a imagem 
    #mostra a tela com o nome de imagem 
    cv.imshow("IA de Hand", img)
    #aqui é o tempo de delay 
    cv.waitKey(1)
    
    
    







