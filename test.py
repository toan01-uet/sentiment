from datetime import time
from data_prep import TextPreprocess, stopwords

import time, pickle
start_time = time.time()
if __name__ == '__main__':
    text = "hom nay la ngay dep troi toi den trường đại học"
    ## preprocessing
    tp = TextPreprocess()
   
    # load the model from disk
    stack_model = pickle.load(open(r'models/stacking_mlxtend.sav', 'rb'))
    print(tp.preprocess(text, stopwords))
    # tfidf
    tfidf_svd = pickle.load(open(r"models/tfidf.pkl", "rb"))
    
    vec_of_text = tfidf_svd.transform([tp.preprocess(text, stopwords)])
     ## predict
    print(stack_model.predict(vec_of_text))
    print("--- %s seconds ---" % (time.time() - start_time))