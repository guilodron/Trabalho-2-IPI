import cv2 as cv
import numpy as np

# Questao 2
cookies = cv.imread('images/cookies.tif', cv.IMREAD_GRAYSCALE)
cv.imshow('Imagem Original', cookies)
thresh_cookie, cookie_binario = cv.threshold(cookies, 0, 255, cv.THRESH_OTSU)
cv.imshow('Cookies binario', cookie_binario)

# Eliminando a cookie mordida
kernel3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (65,65))
erode = cv.erode(cookie_binario, kernel3, iterations=2)
cv.imshow('Erosao', erode)

# Recuperando formato cookie inteiro
dilate = cv.dilate(erode, kernel3, iterations=2)
cv.imshow('Recuperando formato', dilate)

# Máscara na imagem original
cookie_inteiro = cv.bitwise_and(dilate, cookies)
cv.imshow('Mascara Final', cookie_inteiro)

cv.waitKey()
cv.destroyAllWindows()