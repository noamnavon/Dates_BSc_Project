import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
import joblib


def select_features(df, available_features):
    print("\nAvailable features:")
    for idx, feature in enumerate(available_features):
        print(f"{idx}: {feature}")
    selected_input = input("\nEnter the feature numbers (separated by commas or spaces): ")
    selected_input = selected_input.replace(',', ' ')
    selected_indices = [int(i.strip()) for i in selected_input.split() if i.strip().isdigit()]
    if not selected_indices:
        raise ValueError("No features selected!")
    selected_features = [available_features[i] for i in selected_indices]
    selected_features.append('Suitable_For_Treatment')
    return df[selected_features]

def run_svm(selected_df):
    X = selected_df.drop('Suitable_For_Treatment', axis=1)
    y = selected_df['Suitable_For_Treatment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train, y_train)
    joblib.dump(model, "../models/svm_model.pkl")
    y_pred = model.predict(X_test)
    show_results(y_test, y_pred, title="SVM")

def run_plsr(selected_df):
    X = selected_df.drop('Suitable_For_Treatment', axis=1)
    y = selected_df['Suitable_For_Treatment'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    max_components = min(15, X_train.shape[1])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    for n in range(1, max_components + 1):
        rmse_fold = []
        for train_idx, val_idx in skf.split(X_train_scaled, y_train):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            pls = PLSRegression(n_components=n)
            pls.fit(X_tr, y_tr)
            y_pred_val = pls.predict(X_val)
            rmse_fold.append(np.sqrt(mean_squared_error(y_val, y_pred_val)))
        avg_rmse = np.mean(rmse_fold)
        rmse_scores.append(avg_rmse)
    optimal_n = np.argmin(rmse_scores) + 1
    print(f"\nOptimal number of PLS components: {optimal_n}")
    pls_final = PLSRegression(n_components=optimal_n)
    pls_final.fit(X_train_scaled, y_train)
    y_pred_continuous = pls_final.predict(X_test_scaled)
    joblib.dump(pls_final, "../models/plsr_model.pkl")
    y_pred = np.round(y_pred_continuous).astype(int).flatten()
    y_pred = np.clip(y_pred, 0, 2)
    show_results(y_test, y_pred, title="PLSR")

def run_random_forest(selected_df):
    X = selected_df.drop('Suitable_For_Treatment', axis=1)
    y = selected_df['Suitable_For_Treatment'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    joblib.dump((rf, scaler), "../models/random_forest_model.pkl")
    y_pred = rf.predict(X_test_scaled)
    show_results(y_test, y_pred, title="Random Forest")

def run_ann(selected_df):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.utils import to_categorical

    X = selected_df.drop('Suitable_For_Treatment', axis=1)
    y = selected_df['Suitable_For_Treatment']
    y_categorical = to_categorical(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = Sequential()
    model.add(Dense(254, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)
    model.save("../models/ann_model.h5")
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    show_results(y_true, y_pred, title="ANN")

# --- function to display the results ---
def show_results(y_true, y_pred, title="Model"):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print("\nConfusion Matrix:\n", cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"RMSE: {rmse:.4f}")
    plt.figure(figsize=(3, 2))
    from matplotlib.colors import LinearSegmentedColormap
    gray_to_blue = LinearSegmentedColormap.from_list("grayblue", ["#ddddee", "blue"])
    sns.heatmap(cm, annot=True, fmt='d', cmap=gray_to_blue, xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {title}")
    plt.tight_layout()
    plt.show()

def train():
    df = pd.read_csv("../data/all_data.csv")

    available_features = [
        'Fresh_Weigth_Whole_Fruit_(gr)', 'Fresh_Weight_Half_Pericarp_(gr)',
        'Fresh_Weight_Seed_(gr)', 'Dry_Weight_Half_Pericarp_(gr)',
        'Dry_Weigth_Seed_(gr)', 'Cal._Water_Content_Pericarp_(%)',
        'Cal._Water_Content_Seed_(%)', 'Cal._Fresh_Weigth_Pericarp_(gr)',
        'Cal._Dry_Weigth_Pericarp_(gr)', 'TSS_For_Spectral_Analysis_(%)',
        'L:W_Ratio', 'Width_(mm)', 'Length_(mm)', 'cv_width', 'cv_height',
        'cv_mean_L', 'cv_mean_a', 'cv_mean_b', 'cv_var_L', 'cv_var_a', 'cv_var_b'
    ]

    # ---classifiers menu ---

    print("\nSelect classifier:")
    print("1 - SVM")
    print("2 - PLSR")
    print("3 - Random Forest")
    print("4 - ANN")

    classifier_choice = input("\nEnter your choice (1-4): ").strip()

    selected_df = select_features(df, available_features)
    print(selected_df)

    if classifier_choice == '1':
        run_svm(selected_df)
    elif classifier_choice == '2':
        run_plsr(selected_df)
    elif classifier_choice == '3':
        run_random_forest(selected_df)
    elif classifier_choice == '4':
        run_ann(selected_df)
    else:
        print("Invalid choice. Please select 1-4.")