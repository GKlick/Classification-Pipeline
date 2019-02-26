# Classification-Pipeline
Most recent example of Classification Pipeline.


Pipeline created for classification modeling. Pipeline has two distinct parts:
  1. Model creation - applied to training data<br/>
    a. GridSearch - returns best parameters<br/>
    b. Model fitting and scoring - returns model<br/>
    c. Cross Validation - prints error metrics for user to review<br/>
    d. Confusion Matrix - prints confusion matrix  and precision score for user to review<br/>

  2. Model validation - applied to validation data so user can visually see the best threshold for each of the selected models<br/>
    a. Precision Recall graph - interactive plotly graph<br/>
    b. ROC curve - interactive plotly graph<br/>

Note :  For production purposes step two could be automated by applying the models in step one to the validation data and comparing F1 and ROC scores to check for overfitting and thresholding. The visual representation is more for personal use.
