Product Category Classification Project
Salut! Acesta este proiectul meu pentru clasificarea automată a produselor în categorii folosind doar titlul produsului. Mai jos descriu pașii pe care i-am urmat și cum am implementat totul.

1. Structura proiectului
product-category-project/
│
├─ data/
│   └─ products.csv            # datasetul original cu produsele
├─ model/
│   └─ sentiment_model.pkl     # modelul salvat după antrenare
├─ src/
│   ├─ train_model.py          # scriptul de antrenare
│   └─ test_model.py      # scriptul pentru testarea manuală a modelului
└─ README.md

2. Descrierea datasetului
Datasetul conține peste 30.000 de produse, fiecare cu următoarele coloane:
product_id – identificator unic
product_title – titlul produsului
merchant_id – ID-ul vânzătorului
category_label – categoria produsului (targetul nostru)
product_code – cod intern
number_of_views – numărul de vizualizări
merchant_rating – evaluarea vânzătorului
listing_date – data listării

3. Pașii pe care i-am urmat
3.1 Curățarea datelor
Am eliminat rândurile cu valori lipsă.
Am redenumit coloanele datasetului astfel încât să fie cu litere mici și spațiile să fie înlocuite cu _.
Am păstrat doar coloanele relevante: product_title și category_label.
3.2 Pregătirea feature-ului și label-ului
X = product_title
y = category_label
Am folosit doar titlul produsului ca feature, exact cum cere tema.
3.3 Preprocesare
Am aplicat TF-IDF vectorization pe product_title pentru a transforma textul în valori numerice.
3.4 Alegerea și antrenarea modelului
Am folosit RandomForestClassifier cu 200 de estimatori și random_state=42.
Modelul a fost antrenat pe întreg datasetul.
3.5 Salvarea modelului
Am salvat modelul în folderul model din proiect, folosind calea relativă la scriptul train_model.py.
Astfel, modelul poate fi folosit ulterior indiferent de directorul curent de lucru.

4. Testarea manuală
Am creat un script predict_product.py în src/ care permite:
Introducerea unui titlu de produs
Primirea imediată a categoriei prezise de model
Exemplu de utilizare:
python src/predict_product.py
Output-ul va fi:
Enter product title: iphone 7 32gb gold
Predicted category: mobile phones
----------------------------------------
Poți scrie exit pentru a ieși din script.

5. Cum rulez proiectul
1.Clonez sau descarc proiectul.
2.Mă asigur că am instalat dependințele:
pip install pandas scikit-learn joblib
3.Rulez scriptul de antrenare:
python src/train_model.py
4.Testez modelul manual:
python src/predict_product.py

6. Observații
Modelul folosește doar titlul produsului pentru a prezice categoria, fără alte informații despre produs.
Am folosit TF-IDF pentru a transforma textul în valori numerice și RandomForest pentru clasificare.
Toate căile sunt relative la proiect, deci modelul și scripturile pot fi rulate de oriunde.
Proiectul este scris la persoana I, ca să fie clar ce am făcut eu și cum am implementat fiecare pas.
