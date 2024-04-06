"""
detect est une classe pour avoir acces aux donnée nécessaires d'un cheque et les 
fonction usuelle pour les manipuler(les fonctions sont expliquées à lintérieur)
@author: FEIZABADI , ABDOLMOTALLEBI
"""
import numpy as np
from sklearn.datasets import fetch_openml 
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier 
from skimage.transform import resize 
from copy import deepcopy
import matplotlib.image as img 
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from skimage.feature import hog 
class detecte:
    def __init__(self,name_cheque,activation="relu",k_voisin = 3,pic1=np.array([0]),pic2=np.array([0]),pic3=np.array([0])):
        """
        ici on initialise toutes les donnée à partir du nom de l'image ainsi que
        les bases d'entrainement de mnist,la partie de date , de montant ' de numéro de compte
        et de numéro de cheque;
        en plus il y a aussi des paramètre facultatif qu'on aura besoin mais ils sont initialisés
        par défaut et on a droit de les changer si il y en a besoin
        (après avoir testé avec les différante méthode on a bien compris que notre système fonctionne
         mieux à partir des données HOG donc ici on initialise aussi les données nécessaire en HOG)
        """
        
        self.pic1=pic1;self.pic2=pic2;self.pic3=pic3
        self.X,self.y=fetch_openml(name='mnist_784',version=1,return_X_y=True)
        self.X=self.X.values
        self.y=self.y.values  
        self.x_train_hog=self.hog(self.prepa_train())
        self.image=img.imread(name_cheque)
        self.image=np.mean(self.image,2) 
        self.image=1-self.image
        self.activation=activation
        self.n_voisin=k_voisin
        self.val=self.sup_euro(self.jet_vir(self.couper2(self.por(self.image[152:210 ,850:1120]))))
        self.val_hog=self.hog(self.val)
        self.date=self.sup_mom(self.couper2(self.por(self.image[260:290 ,860:1100])))
        self.date_hog=self.hog(self.date)
        self.ch_number=self.checkno()
        self.ch_number_hog=self.hog(self.ch_number)
        self.co_number=self.account()
        self.co_number_hog=self.hog(self.co_number)
        self.spe=img.imread('lesz.jpg')
        self.spe= np . mean (self.spe ,2)
        self.spe=(1-(self.spe/255))
        self.eur=self.por(resize(self.spe[0:112 ,315:427],(28,28)))
        self.vir=self.por(resize(self.spe[0:112 ,440:552],(28,28)))
        self.spe=list()
        for i in range(4000):
            self.spe.append(np.reshape(self.eur,(784,)))
            self.spe.append(np.reshape(self.vir,(784,)))
        self.spe=np.array(self.spe)
        np.random.shuffle(self.spe)
    
    
    
    def hog(self,images):
        """
        cette fonction est suplémentaire et qui prend en argument une liste des images et les renvoit
        sous la forme HOG
        """
        liste=list()
        for img in images:
            liste.append(hog(np.reshape(img,(28,28))))
        liste=np.array(liste)
        return liste
    
    def coupedate(self,elem): 
        """
        c'est une fonction suplémentaire qui prend en paramètre une image et le coupe en 8 partie
        et les renvois dans une liste.
        (on en aura besoin pour couper le numéro de compte et le numéro chèque)
        """
        xdim=0
        u1=[]
        boucle=round(len(elem[1])/8)
        for i in range(boucle):
            u1.append(self.cadre(elem[0:18,xdim:(xdim+8)]))
            xdim+=8
        
        return u1
    def account(self):
        """
        dans cette fonction en utilisant la méthode coupdate on sépare chaque numéro dans le
        numéro de compte à partire de l'image de cette objet qui est déja initialisé et les met
        dans une liste et la renvois'
        """
        
        account1=self.por(self.image[254:272 ,410:705])#pixel de la cadre
        
        """
        ce sont les différantes partie de n_compte dans une chèque
        """
        partie1=account1[:,12:54]
        partie2=account1[:,57:99]
        partie3=account1[:,102:120]
        partie4=account1[:,123:180]
        partie5=account1[:,185:203]
        partie6=account1[:,205:223]
        partie7=account1[:,226:285]
        parties=[partie1,partie2,partie3,partie4,partie5,partie6,partie7]
        res=[]
        for i in parties:
            res+=self.coupedate(i)
        figure, axis=plt.subplots(1,len(res),figsize=(20,20))
        for i in range(len(res)):
            axis[i].imshow(self.cadre(res[i]), cmap='gray')
        return res
        
    def por(self,image,coef=0.4):
        """
        (en effet on a décidé de mettre les valeur de tous les pixels de toutes nos soit en 0 
        soit en 253(en fonction de leurs valeurs initials) , pour qu'on ai une ressemblance entre
        les base d'entrainement et les donnée qu'on veut manipuler)
        c'est une méthode suplémentaire pour mettre les valeurs des pixels soit en 253 soit en 0
        qui prend en paramètre une image,le modifie et le r'envois(il y a des effets de bord)
        """
        for i in range(len(image)):
            for j in range(len(image[0])):
                if image[i][j]>coef:
                    image[i][j]=253
                else: image[i][j]=0
        return image
    def checkno(self):
        """
        cette fonction en utilisant la méthode por renvois tous les image de numéros dans numéros de cheque 
        à partir de l'image de cet objet dans une liste(les image seront modifier avec la méthode por)
        """
        checkno1=self.image[460:490 ,10:710]
        checkno1=self.por(checkno1,coef=0.1)
        u1=self.couper2(checkno1)
        figure, axis=plt.subplots(1,len(u1),figsize=(20,20))
        for i in range(len(u1)):
            axis[i].imshow(self.cadre(u1[i]), cmap='gray')
        return u1
    
    def prepa_train(self):
        """
        dans cette fonction l'idée c'est qu'on modifie la base d'entrainement 
        de telle façon qu'elle ressemble aux images qu'on veux prédire
        en effet comme on a changer les pixel et les mis en 253 on doit entrainer 
        nos classifier avec une base d'entrainement avec les valeurs 0 et 253
        """
        x_train=list()
        for i in self.X:
            """
            ici on met 253 les pixels qui son plus de 170
            et 0 pour les autres
            """
            l=list()
            for j in i:
                if j<190:
                    l.append(0.0)
                else: l.append(253.0)
            l=np.array(l)
            x_train.append(l)
        x_train=np.array(x_train)
        return x_train
        
    def z(self,pic1=np.array([0]),pic2=np.array([0]),pic3=np.array([0])):
        """
        dans cette fonction l'idée c'est qu'on prépare une base d'entrainement avec 
        les image malcentre pour qu'on puisse les supprimer;
        en eefet ici on prépar une base d'entrainement avec les faux images avec leurs étuiquettes en utilisant 
        la fonction roll et random.shuffle de numpy
        (les paramètre facultatif existe pour qu'on puisse ajouter les faux image à partir des
         image des caratères spéciaux comme £')
        """
        y_train=np.array(self.y)
        x_train=self.prepa_train()
        liste_z=list()
        y_z=np.array(['z' for j in range(7000)])
        y_train=np.concatenate((y_train,y_z))
        for i in x_train[:7000]:
            rd=np.random.randint(7,21)
            img=np.reshape(i,(28,28))
            liste_z.append(np.reshape(np.roll(img,rd),(784,)))
        liste_z=np.array(liste_z)
        x_train=np.concatenate((x_train,liste_z))
        """
        si on passe des paramètre facultatif,on rentre dans ces block de code pour les ajouter a 
        notre base d'entrainement
        """
        

        if pic1.any():
            vir=list()
            for i in range(1000):
                rd=np.random.randint(7,21)
                vir.append(np.reshape(np.roll(pic1,rd),(784,)))
            vir=np.array(vir)
            x_train=np.concatenate((x_train,vir))
            y_z=np.array(['z' for j in range(1000)])
            y_train=np.concatenate((y_train,y_z))    
        if pic2.any():
            vir=list()
            for i in range(1000):
                rd=np.random.randint(7,21)
                vir.append(np.reshape(np.roll(pic2,rd),(784,)))
            vir=np.array(vir)
            x_train=np.concatenate((x_train,vir))
            y_z=np.array(['z' for j in range(1000)])
            y_train=np.concatenate((y_train,y_z)) 
        if pic3.any():
            vir=list()
            for i in range(1000):
                rd=np.random.randint(7,21)
                vir.append(np.reshape(np.roll(pic3,rd),(784,)))
            vir=np.array(vir)
            x_train=np.concatenate((x_train,vir))
            y_z=np.array(['z' for j in range(1000)])
            y_train=np.concatenate((y_train,y_z))  
              
        seed=np.random.get_state()
        np.random.shuffle(x_train)
        np.random.set_state(seed)
        np.random.shuffle(y_train)
        return x_train , y_train
    def cadre(self,image):
            """
            cette méthode est suplémentaire et il prend en paramètre un image et ajout un cadre 
            noir autour de ca et le mettre en (28,28) et le renvois
            en effet l'idée c'est mettre le contenant de limage juste au centre
            """
            n=len(image[0])
            img=[]
            image1=np.array([np.array([0.0 for i in range(n)])for j in range(1)])
            image1=np.concatenate((image1,image))
            for li in range(len(image1)):
                l=image1[li]
                zeros=np.array([0.0 for i in range(1)])
                l=np.concatenate((zeros,l,zeros))
                img.append(l)
            img=np.array(img)
            img=np.concatenate((img,np.array([np.array([0.0 for i in range(n+2)]) for j in range(3)])))
            img=resize(img,(28,28))
            for i in range(28):
                for j in range(28):
                    if img[i][j]>90.0:
                        img[i][j]=253.0
                    else: img[i][j]=0.0
       
            return img
    def couper_val(self,images,n=20,pic1=np.array([0]),pic2=np.array([0]),pic3=np.array([0])):
        """
        dans cette fonction l'idée c'est qu'on récupère les bon image
        en effet en utilisan la fonction z on récupère une base d'entrainement avec les 
        faux image et on définit un KNeighborsClassifier et l'entraine avec cette base 
        d'entrainement et pour chaque image on vérifie si la prédiction n'est pas 'Z'
        on garde l'image sinon on le laisse
        (         mettre a jour: aprés d'avoir essayer de supprimer les charecter inutile comme euro
                 en les ajoutant dans notre base d'entrainment de z on a vu qu'il n'a pas marche parce que la base  
                 de z ete beaucoupe et le classifiur a detecté les numére avec éticquette de Z)
        """
        
        
        """
        d'abord en utilisant la méthode fenêtre glissant à l'aide d'un classifier KNN que l'on 
        entraine avec une base de donnée ayant aussi les faux images on prends les image correcte '
        """
        x_train, y_train=self.z(pic1,pic2,pic3)
        clf1=KNeighborsClassifier(n_neighbors=3)
        clf1.fit(x_train,y_train)
        
        x_0=0
        x_1=20
        correcte=list()
        for i in range(n):
            image=images[:,x_0:x_1]
            image=self.cadre(image)
            image=np.reshape(image,(784,))
            m=clf1.predict([image])[0]
            if 'z'!= m :
                correcte.append(image)
                x_0+=15
                x_1+=15
            else:
                x_0+=1;x_1+=1
        """
        ici en utilisant un OneClassSVM avec un kernel poly qui convient bien à nos donnée
         avec son degrès par défaut ,on verifi si il y encore
        des faux images et les supprimer
        en effet en choisissant un limite pour le score de notre classifier on peut connaitre 
        les faux images 
        """
        correcte=np.array(correcte)
        clf1=OneClassSVM(kernel='poly',coef0=0.1)
        x_train=self.prepa_train(self.X)
        clf1.fit(x_train)
        s=clf1.score_samples(correcte)
        correcte=correcte[s>76]
            
        return correcte
    def couper2(self,images):
        """
        en effet on a testé avec beacoup de différants paramètre la fonction coup_val
        ,mais il y avais 20 pourcent d'erreurs pour trouver les bon image,en plus on a pas pus
        retirer le signe £;
        parcontre, on a decidé de fairem autrement; ici on commence collonne par collonne
        une fois arrivé au début d'un chiffre on compte jusqu'à ce il y a plus de
        pixel valant un nombre plus que 0 et on coupe la cette partie et en utlisant 
        la méthode cadre on met les chifres au centre et en 28*28
        puis en utilisant un OneClasseSVM qui seras entrainé cette fois ci avec 
        des image des caractère spéciau pour qu'on puisse les connaitre et les supprimer
        et enfin on renvoit une liste contenant de tous ces images
        """
        image=list()
        i=0
        while i <(len(images[0])):
            condition=False
            ligne=images[:,[i]]
            ligne=np.reshape(ligne,(len(images)))
            ligne=sum(ligne)
            if ligne>0:
                condition=True
                n0=i-2
                while condition:
                    if ligne==0:
                        image.append(images[:,n0:i])
                        condition=False
                    ligne=images[:,[i]]
                    ligne=np.reshape(ligne,(len(images)))
                    ligne=sum(ligne)
                    i+=1
            i+=1
            
        figure, axis=plt.subplots(1,len(image),figsize=(20,20))
        for i in range(len(image)):
            axis[i].imshow(self.cadre(image[i]), cmap='gray')
        for i in range(len(image)):
            image[i]=self.cadre(image[i])
        
        return image
    def sup_mom(self,images):
        """
        on a beaucoup testé les différante méthode por supprimer les caractère spéciaux mais cela 
        augmente le taux d'erreur car le système fonctionne bien quand tous les composants font bien leurs travail
        donc on a decidé de supprimer les caractères qui sont fixes manuellement 
        """
        """
        ici on supprime les slash dans la date qui sont toujours dans les indices 2 et 5
        """
        images.pop(2)
        images.pop(4)
        return images
    def sup_euro(self,images):
        """
        ici on supprime la signe £ qui est toujours dans le dérnier indice 
        """
        a=deepcopy(images)
        a.pop()
        return a
        

    def jet_vir(self,images):  
        """
        par contre pour le virgule comme il n'est pas fixe on a décidé de le trouver à l'aide d'un 
        classifier OneClasseSVM 
        en effet après testé plusieurs fois on a constaté quand on prend les scores à l'aide de la méthode
        score_samples() le minimum est toujours pour les virgule donc on fixe le minimum comme la limite
        et on renvois tous sauf ce qui est à la même indice que le minimum score'
        """
        x_train=self.prepa_train()
        x_train=x_train[:10000]
        clf=OneClassSVM(kernel='poly',coef0=0.3)
        clf.fit(x_train)
        for i in range(len(images)):
            images[i]=self.cadre(images[i])
            images[i]=np.reshape(images[i],(784,))
        images=np.array(images)
        score=clf.score_samples(images)
        limit=min(score)
        images=images[score>limit]
        images=list(images)
        for i in range(len(images)):
            images[i]=np.reshape(images[i],(28,28))
        print(score)
        
        return images
    def k_voisin(self,liste):
        """
        dans cette fonction l'idée c'est qu'on recupère les n proches voisins de 
        chaque image 
        """
        x_train=self.x_train_hog#on prépare la base d'entrainement qu'on a vu tout à l'heur
        clf=KNeighborsClassifier(n_neighbors=self.n_voisin)#definir un classifier KNN avec nvoisin
        clf.fit(x_train,self.y)#entrainement 
        
        voisins=list()
        for image in liste:
            """
            ici pour chaque image on recupère les n proches voisin et  mettre leurs étiquette dans une 
            liste et à la fin on les met tous dans une autre liste avec les même indice 
            que leurs images
            """
            if len(image)==28:
                img=np.reshape(image,(784,))
            else:img=deepcopy(image)
            voisin=list()
            liste_voisin=clf.kneighbors([img],return_distance=False)#récupére les voisin de l'image mais on n'a pas besoin de leur distancs donc on met false "return_distance"
            for v in liste_voisin[0]:
                """
                ici on evite de répeter les etuiquette identique  (c_à_d qu'on veux seulement les différant classes qui sont les classes proches 
                                                                   et on ne s'intéresse pas juste aux proche voisin ')
                """
                if self.y[v] not in voisin:
                    voisin.append(self.y[v])
            voisins.append(voisin)
        return voisins
    def prediction(self,liste):
        """
        dans cette fonction l'idée c'est qu'on predit mais comme on a les plus 
        proches voisin on peux entrainer notre classifier avec juste ces classe et c'est 
        le système neuronal qui va décider entre ces classes 
        en outre on vérifie avant tout qu'il existe plus d'une proche classe et si 
        il y en a une, on la met en tant que la prédiction de l'image
        """
        voisins=self.k_voisin(liste)
        x_train1=self.x_train_hog#récuperer la base d'entrainement corrigée
        pred=list()
        """
                 ici on défini un classifier neuronal et on augmente le nombre de couche mais 
                 on réduit le nombre de neurons dans chaque couche car ça fonctionne beaucoup plus 
                 mieux en plus sur cette base de deonnée le mode  logistic(mais après avoir décide d'utiliser les données sous la forme hog on a la mode d'activation en relu)est la meilleurs 
                 et on utilise l'algorithme gradient descendent avec le learning_rate 'adaptive' 
                 pour qu'il se  diminue le cas où il y en a besoin
                 en fin comme on veux entrainer notre classifier plusieur fois on utilise 
                 le mode WARM_START pour qu'il garde à chaque fois ces algorithmes (mais après avoir tester le programme on a compris que pour utiliser ce mode
                 d'emplois il faut qu'a chaque fois on entraine le classifieur avec les même class donc 
                 on a mis ce mode égale a False)
        """
        clf=MLPClassifier(hidden_layer_sizes=(15,7,9),activation=self.activation,\
                              solver='sgd',alpha=0.001,learning_rate="adaptive",\
                         learning_rate_init=0.01,shuffle=True,warm_start=False,verbose=True,max_iter=50)
       
        n=0
        
        for image in liste:
        
            """     
                ici pour chauque image on récupère les plus proches classes et on vérifie 
                on vérifie si il existe qu'une classe on le considère comme la prédiction
                ,sinon on récupère les donnés des proches classes de la base d'entrainement 
                et on entraine le classifier avec ces données et là le systeme neuronal 
                décide entre les plus proches voisin et nous renvois la prédiction
            
            """
        
            voisin=voisins[n]#récuperer les plus proches classes
            if len(voisin)>1:#verifier qu'il existe plus d'une classe 
                x_train=np.array([[0 for i in range(len(x_train1[0]))]])
                y_train=np.array(['0'])
                """
                ici on récupère les donnée de ces classes et les concatène et
                en utilisant le module numpy on mélange la base d'entrainement et leurs 
                étiquettes avec le même algorithme 
                """
                
                for i in voisin:
                    x_train=np.concatenate((x_train,x_train1[self.y==i]))
                    y_train=np.concatenate((y_train,self.y[self.y==i]))
                seed=np.random.get_state()
                np.random.shuffle(x_train)
                np.random.set_state(seed)
                np.random.shuffle(y_train)
                clf.fit(x_train,y_train)#l'entrainement de classifier
                if len(image)==28:
                    image=np.reshape(image,(784,))
                pre=clf.predict([image])[0]#prédiction
                pred.append(pre)
            elif len(voisin)==1:
                pre=voisin[0]
                pred.append(pre)#on ajoute la prédiction à la liste qu'on veux comme le résultat final
            
            n+=1
          
        return pred#retourner la liste des prédiction
    def data(self):
        date=self.date_hog
        val=self.val_hog
        ch_num=self.ch_number_hog
        co_num=self.co_number_hog
        liste=list()
        for data in [val,date,ch_num,co_num]:
            liste.append(self.prediction(data))
        return liste[0],liste[1],liste[2],liste[3]