
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 18:46:32 2022

@author: rredi
"""
import numpy as np
import torch
import cv2
from torchvision import transforms
from PIL import Image

# Definice transformace
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),  # Zmena veľkosti obrázkov na 28x28
    transforms.ToTensor(),        # Konverzia obrázkov na tenzory
    transforms.Normalize((0.5,), (0.5,)),  # Normalizácia obrázkov na rozsah [-1, 1]
])

def DataPreprocessing(inputData):
    """
    Funkce slouzi pro predzpracovani dat, ktera slouzi k testovani modelu. Veskery kod, ktery vedl k nastaveni
    jednotlivych kroku predzpracovani (vcetne vypoctu konstant, prumeru, smerodatnych odchylek, atp.) budou odevzdany
    spolu s celym projektem.

    :parameter inputData:
        Vstupni data, ktera se budou predzpracovavat.
    :return preprocessedData:
        Predzpracovana data na vystup
    """
    # Odstranění nadbytečných dimenzí
    inputData = np.squeeze(inputData)

    # Kontrola vlastností vstupních dat
    # print("Debugging vstupních dat:")
    # print("Tvar:", inputData.shape)
    # print("Typ:", inputData.dtype)
    # print("Min a Max hodnota:", inputData.min(), inputData.max())

    # Převod na uint8
    if inputData.dtype != np.uint8:
        inputData = (inputData * 255).astype(np.uint8)

    # Převedení numpy pole na PIL Image
    image = Image.fromarray(inputData)

    # Aplikace transformace
    preprocessedData = transform(image)

    # Ploché pole (flatten) pro vstup do MLP modelu
    preprocessedData = preprocessedData.view(-1)
    
    # 2. kontrola
    #print(f"Tvar: {preprocessedData.shape}, typ: {type(preprocessedData)}, min/max: {preprocessedData.min(), preprocessedData.max()}")

    return preprocessedData

