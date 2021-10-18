import cv2 as cv
import numpy as np

def bwareaopen(img, min_size, connectivity=8):
        # Find all connected components (called here "labels")
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(
            img, connectivity=connectivity)
        
        # check size of all connected components (area in pixels)
        for i in range(num_labels):
            label_size = stats[i, cv.CC_STAT_AREA]
            
            # remove connected components smaller than min_size
            if label_size < min_size:
                img[labels == i] = 0
                
        return img

cells = cv.imread('images/img_cells.jpg')
gray = cv.cvtColor(cells, cv.COLOR_BGR2GRAY)
cv.imshow('Imagem original', cells)

# Binarizando imagem
thresh_cells, cells_binario = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
cv.imshow('Cells binarizado', cells_binario)

# Preenchendo pixels das celulas
preenchido = bwareaopen(cells_binario, 125)
cv.imshow('Cells preenchido', cells_binario)

cv.waitKey()
cv.destroyAllWindows()

# Invertendo imagem para trabalhar o algoritmo
invertido_preenchido = 255-preenchido

# Eliminando possivel ruido
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(invertido_preenchido,cv.MORPH_OPEN,kernel, iterations = 2)
cv.imshow('Open', opening)

# Dilatando celulas para garantir uma area de fundo
background = cv.dilate(opening,kernel,iterations=3)
cv.imshow('Background', background)

# Calculando distancia euclidiana ate o pixel preto mais proximo
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)

# Aplicando threshold baseado na distancia calculada
ret, foreground = cv.threshold(dist_transform,0.4*dist_transform.max(),255,0)
foreground = np.uint8(foreground)

# Calculando area desconhecida baseado em foreground e background
desconhecido = cv.subtract(background, foreground)
cv.imshow('Foreground', foreground)
cv.imshow('Desconhecido', desconhecido)

cv.waitKey()
cv.destroyAllWindows()

# Calculando marcardores para o watershed
ret, markers = cv.connectedComponents(foreground)
markers = markers + 1
markers[desconhecido==255] = 0
markers = cv.watershed(cells, markers)
cv.imshow('Marcadores', np.uint8(markers))

# Pintando area de marcadores de verde na imagem original
cells[markers == -1] = [0, 255, 0]

cv.imshow('Watershed', cells)
cv.waitKey()
