# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 17:26:15 2021

@author: FEIZABADI,ABDOLMODALLEBI

"""
import detecte_cheque
condition=True
while condition:
    try:
        name=input("entrez le nom du fichier  ex: cheque1.png  \nle !!fichier doit exister dans la même répertoir\n")
        obj=detecte_cheque.detecte(name)
        data=obj.data()
        print("Montant: ",str(data[0]),"\nDate :",str(data[1]),"\nNuméro de cheque :",str(data[2]),"\nNuméro de compte :",str(data[3]) )
        condition=False
    except:
        print( "le fichier n'exite pas")    
        condition=True
"""
on a testé différante méthode qu'on a déja étudié pendant la semestre et enfin 
pour chaque partie du projet on utilise une méthode différante qui convient et on a décidé
d'utiliser un mélange de KNN et le système neuronal pour faire la prédiction; en plus 
dans la classe detecte on laisse aussi les fonctions qu'on a testé mais qu'on a pas eu
un bon résultats.
pour la prédiction on utilise les données sous la forme HOG mais aussi pour couper 
les images on a utilisé aussi les donnée normales.
en gros, le fonctionnement de ce système est qu'il récupère l'image à partir du nom
qu'on lui passe comme argument et en utilisant les méthode de coupage des image qu'on a expliqué 
à l'intérieur de la class, il coupe les photo et une fois qu'un instance sera crée, il 
les liste des images de chaque numéro qu'on a besoin, puis pour la prédiction,d'abord à l'aide
de la méthode k_voisin() on trouve les plus proches voisin et on récupère les classes 
puis dans la fonction prediction() un classifier MLP sera définis avec les paramètre qu'on a expliqué à l'intérieur 
et pour chaque image on vérifie d'abord les class des voisin s'il y en a plusieur on prépare une base d'entrainement
ne contenant que les données des classes voisin et on entraine notre classifier avec cette base 
et puis demande la prédiction
et on a les resultats suivants pour le premier cheque
Montant:  ['2', '0', '3', '5', '3', '2'] 
Date : ['2', '0', '0', '7', '2', '0', '2', '7'] 
Numéro de cheque : ['0', '0', '0', '0', '3', '5', '0', '2', '3', '0', '0', '1', '1', '5', '5', '6', '9', '8', '5', '0', '0', '7', '0', '0', '0', '5', '5', '4', '5', '5'] 
Numéro de compte : ['7', '5', '2', '3', '0', '0', '0', '0', '3', '5', '0', '3', '0', '0', '0', '2', '2', '5', '5', '3', '3', '2', '2', '0', '0', '0', '0', '2', '5', '5']
"""

