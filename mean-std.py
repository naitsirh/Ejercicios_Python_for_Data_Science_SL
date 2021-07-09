
from math import sqrt

lista = [18, 24, 67, 55, 42, 14, 19, 26, 33]
players = [180, 172, 178, 185, 190, 195, 192, 200, 210, 190]


def media(lista):
	s = 0
	for elemento in lista:
		s += elemento
	return s / len(lista)

def varianza(lista):
	s = 0
	m = media(lista)
	for elemento in lista:
		s += (elemento - m) ** 2
	return s / len(lista)

def desviacion_estandar(lista):
	return sqrt(varianza(lista))

def desviaciones_estandares(lista):
	s = 0
	for elemento in lista:
		if (elemento > media(lista) - desviacion_estandar(lista)) and (elemento < media(lista) + desviacion_estandar(lista)):
			s += 1
	return s



print('La media es:')
print(media(players))

print('\nLa varianza es:')
print(round(varianza(players),2))

print('\nLa desviación estándar es:')
print(desviacion_estandar(players))

print('\nElementos con una sola desviación estándar:')
print(desviaciones_estandares(players))