import pandas as pnd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.model_selection import GridSearchCV

class Determinacion:

    def __init__(self):
            self.mensajesTwitter=pnd.read_csv("datas/calentamientoClimatico.csv", delimiter=";")

    #Carga del archivo

    def cargar(self):
        nltk.download('stopwords')
        nltk.download('wordnet')
        #Información sobre la cantidad de observaciones y su contenido
        print(self.mensajesTwitter.shape)
        print(self.mensajesTwitter.head(2))


    #Transformación de la característica Creencia

    def transformacion(self):
        self.mensajesTwitter['CREENCIA'] = (self.mensajesTwitter['CREENCIA']=='Yes').astype(int)
        print(self.mensajesTwitter.head(100))

    #Función de normalización
    @staticmethod
    def fNormalizacion(mensaje):
        mensaje = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', mensaje)
        mensaje = re.sub('@[^\s]+','USER', mensaje)
        mensaje = mensaje.lower().replace("ё", "е")
        mensaje = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', mensaje)
        mensaje = re.sub(' +',' ', mensaje)
        return mensaje.strip()
    def normalizacion(self):
        #Normalización
        self.mensajesTwitter["TWEET"] = self.mensajesTwitter["TWEET"].apply(Determinacion.fNormalizacion())
        print(self.mensajesTwitter.head(10))

    def eliminacion(self):
        #Carga de StopWords
        stopWords = stopwords.words('english')
        #Eliminación de las Stops Words en las distintas frases
        self.mensajesTwitter['TWEET'] = self.mensajesTwitter['TWEET'].apply(lambda mensaje: ' '.join([palabra for palabra in mensaje.split() if palabra not in (stopWords)]))
        print(self.mensajesTwitter.head(10))


    def stemming(self):
        #Aplicación de stemming
        stemmer = SnowballStemmer('english')
        self.mensajesTwitter['TWEET'] = self.mensajesTwitter['TWEET'].apply(lambda mensaje: ' '.join([stemmer.stem(palabra) for palabra in mensaje.split(' ')]))
        print(self.mensajesTwitter.head(10))

    def lematizacion(self):
        #Lematización
        lemmatizer = WordNetLemmatizer()
        self.mensajesTwitter['TWEET'] = self.mensajesTwitter['TWEET'].apply(lambda mensaje: ' '.join([lemmatizer.lemmatize(palabra) for palabra in mensaje.split(' ')]))
        print(self.mensajesTwitter.head(10))

        print("¡Fin de la preparación!")

    def aprendizaje(self):
        #Conjunto de aprendizaje y de prueba:
        X_train, X_test, y_train, y_test = train_test_split(self.mensajesTwitter['TWEET'].values,  self.mensajesTwitter['CREENCIA'].values,test_size=0.2)

        #Creación de la canalización de aprendizaje


        etapas_aprendizaje = Pipeline([('frequencia', CountVectorizer()),
                                        ('tfidf', TfidfTransformer()),
                                        ('algoritmo', MultinomialNB())])


        #Aprendizaje
        modelo = etapas_aprendizaje.fit(X_train,y_train)

        print(classification_report(y_test, modelo.predict(X_test), digits=4))

    #Frase nueva:
    frase = "Why should trust scientists with global warming if they didnt know Pluto wasnt a planet"
    print(frase)

    #Normalización
    frase = normalizacion(frase)

    #Eliminación de las stops words
    frase = ' '.join([palabra for palabra in frase.split() if palabra not in (stopWords)])

    #Aplicación de stemming
    frase =  ' '.join([stemmer.stem(palabra) for palabra in frase.split(' ')])

    #Lematización
    frase = ' '.join([lemmatizer.lemmatize(palabra) for palabra in frase.split(' ')])
    print (frase)

    prediccion = modelo.predict([frase])
    print(prediccion)
    if(prediccion[0]==0):
        print(">> No cree en el calentamiento climático...")
    else:
        print(">> Cree en el calentamiento climático...")



    #------ Uso de SVM ---

    #Definición de la canalización

    etapas_aprendizaje = Pipeline([('frequencia', CountVectorizer()),
                                        ('tfidf', TfidfTransformer()),
                                        ('algoritmo', svm.SVC(kernel='linear', C=2))])


    #Aprendizaje
    modelo = etapas_aprendizaje.fit(X_train,y_train)
    print(classification_report(y_test, modelo.predict(X_test), digits=4))

    #Búsqueda del mejor parámetro C
    parametrosC = {'algoritmo__C':(1,2,4,5,6,7,8,9,10,11,12)}

    busquedaCOptimo = GridSearchCV(etapas_aprendizaje, parametrosC,cv=2)
    busquedaCOptimo.fit(X_train,y_train)
    print(busquedaCOptimo.best_params_)


    #Parámetro nuevo C=1
    etapas_aprendizaje = Pipeline([('frequencia', CountVectorizer()),
                                        ('tfidf', TfidfTransformer()),
                                        ('algoritmo', svm.SVC(kernel='linear', C=1))])

    modelo = etapas_aprendizaje.fit(X_train,y_train)
    print(classification_report(y_test, modelo.predict(X_test), digits=4))