from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np

def evaluate_model(model, X_test, y_test, is_bert=False):
    if is_bert:
        # Assume trainer.evaluate or custom predict
        preds = model.predict(X_test)  # Adjust
    else:
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
    f1_macro = f1_score(y_test, preds, average='macro')
    f1_micro = f1_score(y_test, preds, average='micro')
    precision = precision_score(y_test, preds, average='macro')
    recall = recall_score(y_test, preds, average='macro')
    roc_auc = roc_auc_score(y_test, probs, average='macro', multi_class='ovr')
    print(classification_report(y_test, preds))
    for i, label in enumerate(y_test.columns):
        cm = confusion_matrix(y_test.iloc[:,i], preds[:,i])
        print(f"Confusion Matrix for {label}:\n{cm}")
    # Error analysis
    misclassified = np.where(np.any(preds != y_test.values, axis=1))[0]
    print("Misclassified examples:")
    for idx in misclassified[:5]:
        print(f"Text: {X_test.iloc[idx]}\nTrue: {y_test.iloc[idx]}\nPred: {preds[idx]}")
    return f1_macro, f1_micro, precision, recall, roc_auc

# Usage: evaluate_model(model_lr_en, X_test_tfidf_en, y_test_en)
# Compare TF-IDF vs embeddings by running on different features