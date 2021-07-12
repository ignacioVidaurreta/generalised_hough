import os
from PIL import Image


# for file in os.listdir('./data'):
#     print(file)
#     im = Image.open(f'./data/{file}')
#     bn = im.convert('RGBA')
#     bn.save(f'./data/bn_{file}')

im = Image.open('./data/TenedorRotadoAgrandado.jpg')
smaller = im.resize((46,124))
smaller.show()
smaller.save("TenedorRotado.jpg")