import cv2 as cv
import numpy as np

# Questao 1 item 1
image = cv.imread('images/morf_test.png')
cv.imshow('Imagem Original', image)
# Binarizando a imagem com threshold de 170
val, thresh = cv.threshold(image, thresh=170, maxval=255, type=cv.THRESH_BINARY)
cv.imshow('Imagem Binarizada', thresh)
cv.waitKey()
cv.destroyAllWindows()

# Questao 1 item 2
# Algoritmo Bottom Hat
kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,7))
close = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=5)
bottom_hat = cv.subtract(close, image)

cv.imshow('Operacao fechamento', close)
cv.imshow('Bottom hat', bottom_hat)
cv.waitKey()
cv.destroyAllWindows()

gray = cv.cvtColor(bottom_hat, cv.COLOR_BGR2GRAY)
val, binary = cv.threshold(gray, thresh=55, maxval=255, type = cv.THRESH_BINARY)
cv.imshow('Bottom hat binarizado', binary)
negative = 255-binary
# Invertendo para fundo branco letras pretas
cv.imshow('Fundo branco', negative)

# Realiza erosão para realçar letras prejudicadas
kernel2 = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
erode = cv.erode(negative, kernel2)
cv.imshow('Imagem final', erode)

cv.waitKey()
cv.destroyAllWindows()
