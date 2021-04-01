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
    
    
    

