# Kaggle-Prj-Shopee-Price_Match_Guarantee
Kaggle-Prj-Shopee-Price_Match_Guarantee


-------

## Submission deadline

May 10, 2021 t 11:59 PM UTC

-------

## Task

In this competition, you’ll apply your machine learning skills to build a model that predicts which items are the same products.

-------

## Evaluation

Submissions will be evaluated based on their mean F1 score. 

The mean is calculated in a sample-wise fashion, meaning that an F1 score is calculated for every predicted row, then averaged.


### F1 score
https://en.wikipedia.org/wiki/F-score

-------

## Image + Text Baseline
https://www.kaggle.com/cuimdi/image-text-baseline

### origin: [Unsupervised] Image + Text Baseline in 20min
https://www.kaggle.com/finlay/unsupervised-image-text-baseline-in-20min

- image hash

- image CNN

- title TFIDF

### f1score

      def getMetric(col):
          def f1score(row):
              n = len( np.intersect1d(row.target,row[col]) )
              return 2*n / (len(row.target)+len(row[col]))
          return f1score
    
-------

## Dataset

- efficientnet-repo-1.1.0


-------


## B3 + TFIDF + KNN = BOOM :P
https://www.kaggle.com/muhammad4hmed/b3-tfidf-knn-boom-p


      -----
      
      # Flag to get cv score
      GET_CV = True

      # Flag to check ram allocations (debug)
      CHECK_SUB = False

      df = cudf.read_csv('../input/shopee-product-matching/test.csv')
      
      # If we are comitting, replace train set for test set and dont get cv
      if len(df) > 3:
          GET_CV = False


      # Function to get our f1 score
      def f1_score(y_true, y_pred):

      # Function to combine predictions
      def combine_predictions(row):


      # Function to read out dataset
      def read_dataset():
          if GET_CV:

          else:

      # Function to decode our images
      def decode_image(image_data):
      
      # Function to read our test image and return image
      def read_image(image):      
      
      # Function to get our dataset that read images
      def get_dataset(image):      
      

- Implements large margin arc distance.

### Reference:
https://arxiv.org/pdf/1801.07698.pdf
        
https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py
     

      -----

      # Arcmarginproduct class keras layer
      class ArcMarginProduct(tf.keras.layers.Layer):
      
      # Function to get the embeddings of our images with the fine-tuned model
      def get_image_embeddings(image_paths):   
      
      # Return tokens, masks and segments from a text array or series 
      def bert_encode(texts, tokenizer, max_len=512):
      
      # Function to get our text title embeddings using a pre-trained bert model
      def get_text_embeddings(df, max_len = 70):      
      
      # Function to get 50 nearest neighbors of each image and apply a distance threshold to maximize cv
      def get_neighbors(df, embeddings, KNN = 50, image = True):  
      
      
      df, df_cu, image_paths = read_dataset()
      image_embeddings = get_image_embeddings(image_paths)
      text_embeddings = get_text_embeddings(df)
      gc.collect()
      
      
      
      model = TfidfVectorizer(stop_words=None, binary=True, max_features=25000)
      
      
      preds = []
      CHUNK = 1024*4

      print('Finding similar titles...')
      CTS = len(df_cu)//CHUNK
      if len(df_cu)%CHUNK!=0: CTS += 1
      for j in range( CTS ):
      
      df_cu['oof_text'] = preds
      
      
      # Get neighbors for image_embeddings
      df, image_predictions = get_neighbors(df, image_embeddings, KNN = 100, image = True)
      
      # Get neighbors for text_embeddings
      df, text_predictions = get_neighbors(df, text_embeddings, KNN = 100, image = False)


      # Concatenate image predctions with text predictions
      if GET_CV:
      
      else:


-------


      
      
      
      
      
      
-------



