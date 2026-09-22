from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.dummy import DummyClassifier

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"

TEST_SIZE = 0.3
RAND_STATE = 42
font_path = str(ASSETS_DIR / "NotoSansSC-Regular.ttf")
chinese_font = font_manager.FontProperties(fname=font_path)

# traditional radicals
traditionals = {
    '小', '龍', '聿', '耒', '弋', '辵', '角', '赤', '田', '毛', '勹', '鹿',
    '斗', '風', '鳥', '青', '生', '土', '麥', '木', '女', '衣', '鹵', '龜',
    '黃', '黽', '示', '犬', '高', '己', '糸', '長', '耳', '而', '虍', '日',
    '鬼', '里', '至', '方', '鬲', '舌', '色', '冫', '匕', '黑', '凵', '虫',
    '辛', '米', '夕', '鼻', '辰', '瓜', '弓', '彳', '入', '甘', '見', '戶',
    '隹', '鼎', '廾', '又', '言', '雨', '金', '目', '尢', '非', '几', '禾',
    '艮', '貝', '冖', '首', '广', '車', '酉', '爻', '矛', '舟', '自', '皮',
    '囗', '无', '牙', '士', '髟', '韋', '屮', '工', '气', '彡', '亠', '走',
    '鬥', '山', '瓦', '齒', '厶', '鼠', '身', '川', '齊', '香', '手', '攴',
    '月', '豸', '肉', '用', '二', '臼', '足', '鬯', '母', '龠', '阜', '十',
    '疒', '頁', '行', '食', '丨', '癶', '匸', '殳', '匚', '文', '止', '火',
    '臣', '冂', '厂', '卩', '八', '氏', '豆', '片', '廴', '力', '欠', '竹',
    '丿', '門', '幺', '牛', '支', '干', '艸', '疋', '缶', '亅', '羊', '寸',
    '舛', '馬', '魚', '刀', '禸', '邑', '丶', '夂', '爪', '立', '黍', '飛',
    '卜', '宀', '心', '父', '石', '革', '韭', '黹', '网', '戈', '口', '儿',
    '羽', '矢', '曰', '巾', '皿', '歹', '爿', '比', '水', '隶', '尸', '白',
    '老', '谷', '一', '襾', '玄', '面', '鼓', '玉', '骨', '子', '麻', '音',
    '斤', '釆', '彐', '穴', '乙', '人', '豕', '血', '夊', '大'}

# simplified radicals
simplifieds = {'韦', '麦', '马', '鼡', '齐', '龙', '风', '见', '页', '黾', '讠',
               '⻊', '饣', '飞', '鱼', '长', '车', '鸟', '龟', '齿', '纟', '门',
               '贝', '钅', '户', '⺾', '黄'}


def rf_grid_search(x_train, x_test, y_train, y_test):
    """
    Perform grid search for best Random Forest parameters and print results.
    :param x_train: training data
    :param x_test: test data
    :param y_train: training data labels
    :param y_test: test data labels
    """
    # create model
    rf = RandomForestClassifier(random_state=RAND_STATE)

    # parameters to check
    param_grid = {
        "criterion": ["gini", "entropy"],
        "n_estimators": [100, 200, 500],  # trees
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "class_weight": [None, "balanced"]
    }

    # find best
    grid_search_rf = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,  # cross validation
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2  # show messages
    )

    # use the training data
    grid_search_rf.fit(x_train, y_train)

    # print best results
    print("Best parameters found (Random Forest):",
          grid_search_rf.best_params_)
    print("Best cross-validation score (Random Forest):",
          grid_search_rf.best_score_)

    # predict using best results
    best_rf = grid_search_rf.best_estimator_
    y_pred_rf = best_rf.predict(x_test)

    print("\n---- Best Random Forest Results ----")
    print(classification_report(y_test, y_pred_rf))


def dt_grid_search(x_train, x_test, y_train, y_test):
    """
    Perform grid search for best Decision Tree parameters and print results.
    :param x_train: training data
    :param x_test: test data
    :param y_train: training data labels
    :param y_test: test data labels
    """
    # create model
    clf = DecisionTreeClassifier(random_state=RAND_STATE)

    # parameters to check
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "class_weight": [None, "balanced"]
    }

    # find best
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2
    )

    # use the training data
    grid_search.fit(x_train, y_train)

    # print best results
    print("Best parameters found:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    # predict using best results
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(x_test)

    print("\n---- Best Decision Tree Results ----")
    print(classification_report(y_test, y_pred))


def perform_grid_search():
    """
    Load dataset, prepare features, and run grid search on classifiers.
    Uncomment the functions who perform the search for the different models
    (at the end of this function).
    """
    mlb, df, x, x_train, x_test, y_train, y_test = prepare_data()

    # uncomment to run the search for the wanted model:
    dt_grid_search(x_train, x_test, y_train, y_test)
    # rf_grid_search(x_train, x_test, y_train, y_test)


def best_parameters(x_train, x_test, y_train, y_test):
    """
    Train models with previously found best parameters and evaluate.
    Best parameters found after running the function perform_grid_search().
    :param x_train: training data
    :param x_test: test data
    :param y_train: training data labels
    :param y_test: test data labels
    :return: best_clf, best_rf (DecisionTreeClassifier, RandomForestClassifier)
    """
    # Best parameters found: (0.3)
    # {'class_weight': None, 'criterion': 'gini', 'max_depth': None,
    # 'min_samples_leaf': 2, 'min_samples_split': 5}
    best_clf = DecisionTreeClassifier(random_state=RAND_STATE,
                                      min_samples_leaf=2,
                                      min_samples_split=5)
    best_clf.fit(x_train, y_train)
    y_pred_clf = best_clf.predict(x_test)

    print("\n---- Best Decision Tree Results ----")
    print(classification_report(y_test, y_pred_clf))

    # Best parameters found (Random Forest): (0.3)
    # {'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': None,
    # 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
    best_rf = RandomForestClassifier(class_weight='balanced',
                                     criterion='entropy',
                                     min_samples_split=5, n_estimators=200,
                                     random_state=RAND_STATE)
    best_rf.fit(x_train, y_train)
    y_pred_rf = best_rf.predict(x_test)

    print("\n---- Best Random Forest Results ----")
    print(classification_report(y_test, y_pred_rf))

    return best_clf, best_rf


def user_prediction(clf, df, mlb, x_columns):
    """
    Interactive prediction for a character chosen by row index.
    Uses the function predict_char() for the prediction.
    :param clf: a trained classifier (tree or forest)
    :param df: the datatset (pandas DataFrame)
    :param mlb: MultiLabelBinarizer for encoding the radiclas
    :param x_columns: the features columns
    """
    print("\n\nchoose a random number -> get a prediction for the "
          "corresponding chinese character")
    predict_again = True

    while predict_again:
        while True:
            try:
                idx = int(
                    input(f"choose a number between (0 - {len(df) - 1}): "))
                if idx < 0 or idx >= len(df):
                    print("number out of range! try again")
                    continue
                break
            except ValueError:
                print("please enter a valid integer")

        row = df.iloc[idx]
        print(f"\npredicting the category for the char: {row['char']} ...")

        # predict for the char the user chose
        predict_char(
            clf=clf,
            char=row["char"],
            strokes=row["total_strokes"],
            radical=row["radical"],
            all_radicals=row["all_radicals"],
            mlb=mlb,
            x_columns=x_columns
        )
        print(f"real category: {row['category']}")
        while True:
            user_predict_again = input("predict again? Y/N: ")
            if user_predict_again in {"Y", "N"}:
                predict_again = True if user_predict_again == "Y" else False
                break
            else:
                print("please enter 'Y' for Yes or 'N' for No")


def predict_char(clf, char, strokes, radical, all_radicals, mlb, x_columns):
    """
    Predict category for a single given chinese character using the given
    classifier.
    :param clf: a trained classifier (tree or forest)
    :param char: the char to predict for
    :param strokes: the chars number of strokes
    :param radical: the chars dictionary radical
    :param all_radicals: all the radicals in the char
    :param mlb: MultiLabelBinarizer for encoding the radiclas
    :param x_columns: the features columns
    """
    # new df for char
    new_df = pd.DataFrame([{
        "total_strokes": strokes,
        "radical": radical,
        "all_radicals": all_radicals
    }])

    # create features
    new_df["is_simplified_radical"] = new_df["radical"].astype(
        str).str.contains("'").astype(int)
    new_radicals_matrix = mlb.transform(new_df["all_radicals"])
    new_radical_dummies = pd.DataFrame(new_radicals_matrix,
                                       columns=mlb.classes_)
    x_new = pd.concat([new_df[["total_strokes", "is_simplified_radical"]],
                       new_radical_dummies], axis=1)

    # same as x_columns
    x_new = x_new.reindex(columns=x_columns, fill_value=0)

    # predict for char
    prediction = clf.predict(x_new)[0]
    print(f"the char {char} has been predicted to be: {prediction}")

    # predict probability
    probas = clf.predict_proba(x_new)[0]
    prob_dict = {cls: round(prob, 3) for cls, prob in
                 zip(clf.classes_, probas)}
    print(f"probabilities: {prob_dict}")
    return prediction


def categorize_top_features(top_features):
    """
    Rename feature names for readability.
    :param top_features: the top features and corresponding importances
    :return: the top features, renamed
    """
    display_names = {
        "is_simplified_radical": "is dict. rad. simplified?",
        "total_strokes": "char stroke count"}
    renamed_features = []
    for feat in top_features.index:
        if feat in traditionals and feat in simplifieds:
            renamed_features.append(feat)
        elif feat in traditionals:
            renamed_features.append("rad. " + feat + " (trad. form)")
        elif feat in simplifieds:
            renamed_features.append("rad. " + feat + " (simp. form)")
        else:
            renamed_features.append(display_names[feat])
    top_features.index = renamed_features
    return top_features


def confusion_matrix_viz(model, y_test, y_pred, model_name="Model"):
    """
    Plot and save confusion matrix for model predictions of the given model.
    :param model: a trained classifier (tree or forest)
    :param y_test: test data labels
    :param y_pred: predicted labels
    :param model_name: model name (str)
    """
    if model_name == "Dummy_Decision_Tree":
        subtitle = "Predicts the majority class ('Both')"
    else:
        subtitle = "Good prediction rates for all categories\n Strong " \
                   "separation between Traditional and Simplified, " \
                   "most errors with 'Both' "

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=model.classes_)
    disp.plot(cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.suptitle(subtitle)
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.text(
        1.30, 0.5, "Number of samples",
        rotation=90,
        va="center", ha="left", fontweight="bold",
        transform=plt.gca().transAxes
    )

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / f"CM_{model_name}.png")
    plt.close()


def feature_importances_viz(model, x_train, model_name="Model"):
    """
    Plot and save top feature importances for the given model.
    :param model: a trained classifier (tree or forest)
    :param x_train: training data
    :param model_name: model name (str)
    """
    importances = pd.Series(model.feature_importances_,
                            index=x_train.columns)
    top_features = importances.sort_values(ascending=False).head(10)
    top_features = categorize_top_features(top_features)
    print("\nTop 10 Feature Importances:")
    print(top_features)

    top_features_percent = top_features * 100
    ax = top_features_percent.plot(kind="barh",
                                   figsize=(8, 5),  # 8, 5
                                   title=f"{model_name} - Top 10 Features",
                                   color="skyblue")
    plt.gca().invert_yaxis()
    plt.suptitle("Model relies on character form more than specific radicals")
    plt.xlabel("Importance Score (%)", fontweight="bold")
    plt.ylabel("Feature", fontweight="bold")
    for i, (value, name) in enumerate(
            zip(top_features_percent.values, top_features_percent.index)):
        ax.text(value + 0.5,
                i,
                f"{value:.4f}%",
                va="center",
                fontsize=10)

    plt.xlim(0, 100)  # limit x to 100% importance score
    plt.yticks(fontproperties=chinese_font)  # so we can read the radicals
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / f"FEATURE_IMPORTANCES_{model_name}.png")
    plt.close()


def reports_viz(all_reports):
    """
    Plot and save bar chart to compare macro-avg precision/recall/F1 for all
    models.
    :param all_reports: dict of reports {model_name: macro avg metrics}
    """
    df_reports = pd.DataFrame(all_reports).T
    df_reports = df_reports[["precision", "recall", "f1-score"]]
    df_reports = df_reports.T  # bars = models

    ax = df_reports.plot(kind="bar", figsize=(10, 6))
    plt.title("Model Comparison - Macro Avg Metrics")
    plt.suptitle(
        "Trained classifier models surpass baseline model significantly")
    plt.ylabel("Score", fontweight="bold")
    plt.xlabel("Metric", fontweight="bold")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)  # horizontal

    # legend outside graph (upper left)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")

    # values above bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9, rotation=0)

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "model_comparison.png", bbox_inches="tight")
    plt.close()


def evaluate_model(model, x_train, x_test, y_test, model_name="Model"):
    """
    Evaluate a given model performance with classification report and plots.
    :param model: a trained classifier (tree or forest)
    :param x_train: training data
    :param x_test: test data
    :param y_test: test data labels
    :param model_name: model name (str)
    :return: classification report (dict)
    """
    print(f"\n---- {model_name} Evaluation ----")

    # Predictions
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True,
                                   zero_division=0)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion Matrix
    confusion_matrix_viz(model, y_test, y_pred, model_name)

    # Feature Importances (if model has feature importances)
    if hasattr(model, "feature_importances_"):
        feature_importances_viz(model, x_train, model_name)
    return report


def prepare_data():
    """
    Load and prepare the dataset for classification.
    Create category and features.
    Split the data to into train/test.
    :return: x_train, x_test, y_train, y_test (training and testing data)
    """
    # read data
    df = pd.read_csv(DATA_DIR / "chinese_characters_data3.csv")

    # label: category (Simplified/Traditional/"Both")
    df = df[df["category"].isin(["Simplified", "Traditional", "Both"])]
    y = df["category"]

    # --- get attributes  ---
    # 1. stroke count in df["total_strokes"]
    # 2. dictionary radical classification (trad/simp)
    df["is_simplified_radical"] = df["radical"].astype(str).str.contains(
        "'").astype(int)

    # 3. parse all_radicals and one-hot encode
    df["all_radicals"] = df["all_radicals"].apply(eval)
    mlb = MultiLabelBinarizer()
    radicals_matrix = mlb.fit_transform(df["all_radicals"])
    radical_dummies = pd.DataFrame(radicals_matrix, columns=mlb.classes_,
                                   index=df.index)

    # choose attributes, split data
    x = pd.concat(
        [df[["total_strokes", "is_simplified_radical"]], radical_dummies],
        axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=TEST_SIZE,
                                                        random_state=RAND_STATE,
                                                        stratify=y)
    return mlb, df, x, x_train, x_test, y_train, y_test


def predict_and_evaluate():
    """
    Train baseline, Decision Tree, and Random Forest models.
    Evaluate each model, generate visualizations, and compare metrics.
    :return: best_clf, best_rf, df, mlb, x.columns (DecisionTreeClassifier,
            RandomForestClassifier, dataset DataFrame,
            fitted MultiLabelBinarizer, feature columns)
    """
    mlb, df, x, x_train, x_test, y_train, y_test = prepare_data()

    all_reports = {}  # collect to compare in graph

    # --- BASELINE: DummyClassifier ---
    # always chooses the most frequent label
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    report_ddt = evaluate_model(dummy, x_train, x_test, y_test,
                                model_name="Dummy_Decision_Tree")
    all_reports["Dummy Decision Tree"] = report_ddt["macro avg"]

    # --- BEST DECISION TREE ---
    best_clf = DecisionTreeClassifier(random_state=RAND_STATE,
                                      min_samples_leaf=2,
                                      min_samples_split=5)
    best_clf.fit(x_train, y_train)
    report_dt = evaluate_model(best_clf, x_train, x_test, y_test,
                               model_name="Decision_Tree")
    all_reports["Decision Tree"] = report_dt["macro avg"]

    # --- BEST RANDOM FOREST ---
    best_rf = RandomForestClassifier(class_weight='balanced',
                                     criterion='entropy',
                                     min_samples_split=5, n_estimators=200,
                                     random_state=RAND_STATE)
    best_rf.fit(x_train, y_train)
    report_rf = evaluate_model(best_rf, x_train, x_test, y_test,
                               model_name="Random_Forest")
    all_reports["Random Forest"] = report_rf["macro avg"]

    # --- viz reports ---
    reports_viz(all_reports)

    return best_clf, best_rf, df, mlb, x.columns


if __name__ == "__main__":
    # uncomment to perform grid search:
    # perform_grid_search()

    # build the classifier, predict and evaluate performance:
    best_clf, best_rf, df, mlb, x_columns = predict_and_evaluate()

    # uncomment for interactive prediction:
    # user_prediction(best_rf, df, mlb, x_columns)
