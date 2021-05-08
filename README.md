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

## Submission File

You must create a space-delimited list of all posting_ids that match the posting in the posting_id column. 

Posts always self-match. Group sizes were capped at 50, so there is no benefit to predict more than 50 matches.

The file should have a header, be named submission.csv, and look like the following:

      posting_id,matches
      test_123,test_123
      test_456,test_456 test_789

You should predict matches for every posting_id. For example, if you believe A matches B and C, A,A B C, you would also include B,B A C and C,C A B.


-------

## Data Description

Finding near-duplicates in large datasets is an important problem for many online businesses. 

In Shopee's case, everyday users can upload their own images and write their own product descriptions, adding an extra layer of challenge. 

Your task is to identify which products have been posted repeatedly. 

The differences between related products may be subtle while photos of identical products may be wildly different!

As this is a code competition, only the first few rows/images of the test set are published; the remainder are only available to your notebook when it is submitted. 

Expect to find roughly 70,000 images in the hidden test set. 

The few test rows and images that are provided are intended to illustrate the hidden test set format and folder structure.


-------


## Files

### [train/test].csv

the training set metadata. 

Each row contains the data for a single posting. Multiple postings might have the exact same image ID, but with different titles or vice versa.

- posting_id

the ID code for the posting.

- image

the image id/md5sum.

- image_phash

a perceptual hash of the image.

- title

the product description for the posting.

- label_group

ID code for all postings that map to the same product. Not provided for the test set.

### [train/test]images

the images associated with the postings.

### sample_submission.csv

a sample submission file in the correct format.

- posting_id

the ID code for the posting.

- matches

Space delimited list of all posting IDs that match this posting. 

Posts always self-match. Group sizes were capped at 50, so there's no need to predict more than 50 matches.






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

## Papers

### ArcFace: Additive Angular Margin Loss for Deep Face Recognition
https://arxiv.org/pdf/1801.07698.pdf


### SphereFace: Deep Hypersphere Embedding for Face Recognition
https://arxiv.org/pdf/1704.08063.pdf


### CosFace: Large Margin Cosine Loss for Deep Face Recognition
https://arxiv.org/pdf/1801.09414.pdf


### DeepFace: Closing the Gap to Human-Level Performance in Face Verification
https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf



-------

## Progress

### Public Best LB Score: 0.728

### Private Score:




-------


## B3 + TFIDF + KNN = BOOM :P      by Muhammad Ahmed
https://www.kaggle.com/muhammad4hmed/b3-tfidf-knn-boom-p


      -----

### Flag

      # Flag to get cv score
      GET_CV = True

      # Flag to check ram allocations (debug)
      CHECK_SUB = False

### dataframe

      df = cudf.read_csv('../input/shopee-product-matching/test.csv')
      
      # If we are comitting, replace train set for test set and dont get cv
      if len(df) > 3:
          GET_CV = False

### Function

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
      

### Implements large margin arc distance.

### Reference:
https://arxiv.org/pdf/1801.07698.pdf
        
https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py
     

      -----

      # Arcmarginproduct class keras layer
      class ArcMarginProduct(tf.keras.layers.Layer):
      
          def __init__(self, n_classes, s=30, m=0.50, easy_margin=False, ls_eps=0.0, **kwargs):
          
          def get_config(self):
      
          def build(self, input_shape):
          
          def call(self, inputs):   
         
      -----

### Function

      # Function to get the embeddings of our images with the fine-tuned model
      def get_image_embeddings(image_paths):   
      
      # Return tokens, masks and segments from a text array or series 
      def bert_encode(texts, tokenizer, max_len=512):
      
      # Function to get our text title embeddings using a pre-trained bert model
      def get_text_embeddings(df, max_len = 70):      
      
      # Function to get 50 nearest neighbors of each image and apply a distance threshold to maximize cv
      def get_neighbors(df, embeddings, KNN = 50, image = True):  
      
      -----
      
      df, df_cu, image_paths = read_dataset()
      image_embeddings = get_image_embeddings(image_paths)
      text_embeddings = get_text_embeddings(df)
      gc.collect()
      
      
      -----
      
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




### Public Score

       Public Score: 0.727


-------  

## Eff-B4 + TFIDF >= 0.728
https://www.kaggle.com/vatsalmavani/eff-b4-tfidf-0-728



      -----

### Import Packages

### Config

      class CFG:

### Utils

      def read_dataset():      
          df = pd.read_csv('../input/shopee-product-matching/test.csv')
          df_cu = cudf.DataFrame(df)
          image_paths = '../input/shopee-product-matching/test_images/' + df['image']

      return df, df_cu, image_paths

      def seed_torch(seed=42):
      
      def f1_score(y_true, y_pred):
      
      def combine_predictions(row):


### Image Predictions

      class ArcMarginProduct(nn.Module):

      class ShopeeModel(nn.Module):

      def get_image_neighbors(df, embeddings, KNN=50):

      def get_test_transforms():

      class ShopeeDataset(Dataset):      
      
      def get_image_embeddings(image_paths):
          model = ShopeeModel(pretrained=False).to(CFG.device)
          
          image_dataset = ShopeeDataset(image_paths=image_paths, transforms=get_test_transforms())

### Text Predictions 

      def get_text_predictions(df, max_features=25_000):

### Calculating Predictions

      df,df_cu,image_paths = read_dataset()

      # Get neighbors for image_embeddings
      image_embeddings = get_image_embeddings(image_paths.values)
      text_predictions = get_text_predictions(df, max_features=25_000)
      df, image_predictions = get_image_neighbors(df, image_embeddings, KNN=50 if len(df)>3 else 3)

      -----
      
### class CFG:

       batch_size = 8:       LB 0.728   ver5
       batch_size = 16:      LB 0.728   ver4
       batch_size = 20:      LB 0.728   ver1   --- (best)   #default
       batch_size = 32:      LB 0.728   ver3
       batch_size = 64:      LB error   ver7
       

### df, image_predictions = get_image_neighbors(df, image_embeddings,

batch_size = 20:  

      KNN=50 if len(df)>3 else 3              LB 0.728   ver1    #default
      KNN=100 if len(df)>3 else 3             LB 0.728   ver8   --- best   172 -> 151




-------

## Eff-B4 + TFIDF w/ CV for threshold_searching
https://www.kaggle.com/chienhsianghung/eff-b4-tfidf-w-cv-for-threshold-searching

      Public Score              LB 0.728   ver1      394 -> 396


-------

## TF-IDF + Rapids + Arc Margin Shopee
https://www.kaggle.com/prashantchandel1097/ensemble-of-multiple-models-lb0-733

### Configuration

threshold = 0.3:

      BATCH_SIZE = 8               LB 0.733    ver1      --- best    -- default
      BATCH_SIZE = 12              LB 0.733    ver9 
      BATCH_SIZE = 16              LB 0.733    ver5 
      BATCH_SIZE = 24              LB error    ver8
      BATCH_SIZE = 32              LB error    ver7 
      

### df, image_predictions = get_neighbors()

threshold = 0.3:

      KNN = 25               LB 0.733    ver1      --- best    -- default
      KNN = 50               LB 0.733    ver10      

### image_predictions1

      threshold = 0.25             LB 0.732    ver3 
      threshold = 0.3              LB 0.733    ver1                    --- default
      threshold = 0.31             LB 0.733    ver13      222 -> 222
      threshold = 0.315            LB 0.733    ver14      229 -> 233
      threshold = 0.32             LB 0.733    ver11      453 -> 220    --- best
      threshold = 0.325            LB 0.733    ver15      234 -> 234
      threshold = 0.33             LB 0.733    ver12      220 -> 222
      threshold = 0.35             LB 0.732    ver2  

threshold = 0.32:

      self.dropout = nn.Dropout(p=0.0)             LB 0.733    ver11     --- best
      self.dropout = nn.Dropout(p=0.2)             LB 0.733    ver18         235 -> 235
      self.dropout = nn.Dropout(p=0.5)             LB 0.733    ver16         234 -> 235
      self.dropout = nn.Dropout(p=1.0)             LB 0.733    ver17         235 -> 235

self.dropout = nn.Dropout(p=0.0):

        #IDX = np.where(cts[k,]>0.7)[0]
        
        IDX = cupy.where(cts[k,]>0.70)[0]     LB 0.731    ver19
        IDX = cupy.where(cts[k,]>0.73)[0]     LB 0.733    ver21      248 -> 249
        IDX = cupy.where(cts[k,]>0.74)[0]     LB 0.733    ver22      249 -> 249
        IDX = cupy.where(cts[k,]>0.75)[0]     LB 0.733    ver11                       --- default
        IDX = cupy.where(cts[k,]>0.76)[0]     LB 0.733    ver23      249 -> 226       --- best   
        IDX = cupy.where(cts[k,]>0.77)[0]     LB 0.733    ver20      245 -> 247
        
### def get_test_transforms(): 

      #ToTensorV2(p=1.0)      LB     ver24      234 -> 
      ToTensorV2(p=1.0)      LB 0.733    ver23    --- best       --- default



### class CFG:        

      batch_size =  8   LB     ver     -> 
      batch_size = 12   LB 0.733    ver23         --- best         --- default
      batch_size = 16   LB    ver       -> 
      
      
      
-------

