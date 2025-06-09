import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
import uuid
import random

def calculate_flagging_score(db_name="scoring_database.db", output_db_name="results_database.db"):
    """Calculate flagging scores and statuses for products, saving to a new database."""
    try:
        # Connect to input database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='products'")
        if not cursor.fetchone():
            raise sqlite3.OperationalError("Table 'products' does not exist in the database")

        # Query data
        query = "SELECT product_id, satisfiability_score, image_authenticity_score FROM products"
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            raise ValueError("No data found in the products table")

        # Data cleaning: Remove nulls and clamp scores to 0-100
        df = df.dropna()
        df['satisfiability_score'] = df['satisfiability_score'].clip(0, 100)
        df['image_authenticity_score'] = df['image_authenticity_score'].clip(0, 100)

        # Feature engineering: Add interaction term
        df['interaction_score'] = df['satisfiability_score'] * df['image_authenticity_score']

        # Rule-based scoring: Weighted average (40% satisfiability, 40% authenticity, 20% interaction)
        df['rule_based_score'] = (df['satisfiability_score'] * 0.4 + 
                                 df['image_authenticity_score'] * 0.4 + 
                                 df['interaction_score'] * 0.2 / 100)

        # Create synthetic labels with added noise to avoid trivial mapping
        df['synthetic_label'] = np.where(
            (df['satisfiability_score'] < 30) & (df['image_authenticity_score'] < 30), 
            20 + random.uniform(-5, 5),
            np.where((df['satisfiability_score'] < 50) | (df['image_authenticity_score'] < 50), 
                     50 + random.uniform(-5, 5), 
                     70 + random.uniform(-5, 5))
        ).clip(0, 100)

        # Create synthetic status labels for classification
        df['synthetic_status'] = np.where(
            df['synthetic_label'] <= 30, 'Flagged',
            np.where(df['synthetic_label'] <= 60, 'Sent for Human Review', 'Genuine Product')
        )

        # Prepare features
        X = df[['satisfiability_score', 'image_authenticity_score', 'interaction_score']]
        y_reg = df['synthetic_label']
        y_clf = df['synthetic_status']

        # Split data into train and test sets
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
            X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
        )

        # Apply SMOTE to balance classes for classification
        smote = SMOTE(random_state=42)
        X_train_clf, y_clf_train = smote.fit_resample(X_train, y_clf_train)

        # Train Random Forest Classifier with hyperparameter tuning
        clf_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        clf = GridSearchCV(
            RandomForestClassifier(random_state=42, class_weight='balanced'),
            clf_param_grid,
            cv=5,
            scoring='f1_macro'
        )
        clf.fit(X_train_clf, y_clf_train)

        # Predict on test set only
        df.loc[X_test.index, 'status'] = clf.predict(X_test)

        # Train Random Forest Regressor for flagging score
        reg_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        rf = GridSearchCV(
            RandomForestRegressor(random_state=42),
            reg_param_grid,
            cv=5,
            scoring='r2'
        )
        rf.fit(X_train, y_reg_train)
        df.loc[X_test.index, 'rf_score'] = rf.predict(X_test).clip(0, 100)

        # Combine scores for test set: Weighted average of rule-based and Random Forest
        df.loc[X_test.index, 'flagging_score'] = (0.4 * df.loc[X_test.index, 'rule_based_score'] + 
                                                  0.6 * df.loc[X_test.index, 'rf_score'])

        # Optimize thresholds for status assignment on test set
        thresholds_flagged = [30, 40, 50]
        thresholds_review = [50, 60, 70]
        best_f1 = 0
        best_thresholds = (40, 60)
        for flag in thresholds_flagged:
            for review in thresholds_review:
                if flag < review:
                    temp_status = np.where(df.loc[X_test.index, 'flagging_score'] <= flag, 'Flagged',
                                          np.where(df.loc[X_test.index, 'flagging_score'] <= review, 
                                                   'Sent for Human Review', 'Genuine Product'))
                    f1 = classification_report(y_clf_test, temp_status, 
                                             output_dict=True, zero_division=0)['macro avg']['f1-score']
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresholds = (flag, review)

        # Apply best thresholds to test set
        df.loc[X_test.index, 'status'] = np.where(df.loc[X_test.index, 'flagging_score'] <= best_thresholds[0], 'Flagged',
                                                 np.where(df.loc[X_test.index, 'flagging_score'] <= best_thresholds[1], 
                                                          'Sent for Human Review', 'Genuine Product'))

        # For training set, apply default thresholds (to ensure all rows have values)
        df.loc[X_train.index, 'rf_score'] = rf.predict(X_train).clip(0, 100)
        df.loc[X_train.index, 'flagging_score'] = (0.4 * df.loc[X_train.index, 'rule_based_score'] + 
                                                  0.6 * df.loc[X_train.index, 'rf_score'])
        df.loc[X_train.index, 'status'] = np.where(df.loc[X_train.index, 'flagging_score'] <= 40, 'Flagged',
                                                  np.where(df.loc[X_train.index, 'flagging_score'] <= 60, 
                                                           'Sent for Human Review', 'Genuine Product'))

        # Save results to new SQLite database
        conn = sqlite3.connect(output_db_name)
        df[['product_id', 'satisfiability_score', 'image_authenticity_score', 
            'flagging_score', 'status']].to_sql('results', conn, if_exists='replace', index=False)
        conn.close()

        # Print summary
        print("Status distribution (test set):")
        print(df.loc[X_test.index, 'status'].value_counts())
        print(f"Results saved to {output_db_name} in table 'results'")
        print(f"Best thresholds: Flagged <= {best_thresholds[0]}, Sent for Human Review <= {best_thresholds[1]}")
        return df, X_test.index
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None, None
    except ValueError as e:
        print(f"Data error: {e}")
        return None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None

def evaluate_model_accuracy(db_name="scoring_database.db", results_db_name="results_database.db", test_indices=None):
    """Evaluate the accuracy of the flagging model using ground truth data."""
    try:
        # Load ground truth data
        conn = sqlite3.connect(db_name)
        query = "SELECT product_id, satisfiability_score, image_authenticity_score, true_flagging_score, true_status FROM ground_truth"
        df_ground_truth = pd.read_sql_query(query, conn)
        conn.close()

        if df_ground_truth.empty:
            raise ValueError("No data found in the ground_truth table")

        # Load model predictions
        conn = sqlite3.connect(results_db_name)
        df_predictions = pd.read_sql_query("SELECT product_id, flagging_score, status FROM results", conn)
        conn.close()

        # Merge ground truth and predictions
        df = df_ground_truth.merge(df_predictions, on="product_id")
        df = df.dropna()

        # Filter for test set if indices are provided
        if test_indices is not None:
            df = df[df.index.isin(test_indices)]

        # Classification metrics
        print("\nClassification Metrics for Status (Test Set):")
        print(classification_report(df['true_status'], df['status'], zero_division=0))

        # Regression metrics
        print("\nRegression Metrics for Flagging Score (Test Set):")
        mae = mean_absolute_error(df['true_flagging_score'], df['flagging_score'])
        rmse = np.sqrt(mean_squared_error(df['true_flagging_score'], df['flagging_score']))
        r2 = r2_score(df['true_flagging_score'], df['flagging_score'])
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.2f}")

        # Save evaluation results
        conn = sqlite3.connect("evaluation_results.db")
        df[['product_id', 'satisfiability_score', 'image_authenticity_score', 'true_flagging_score', 
            'true_status', 'flagging_score', 'status']].to_sql(
            'evaluation_results', conn, if_exists='replace', index=False)
        conn.close()
        print("\nEvaluation results saved to evaluation_results.db")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except ValueError as e:
        print(f"Data error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def create_product_database(n_records, db_name="scoring_database.db"):
    """Create a SQLite database with n_records of product data and synthetic ground truth."""
    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Create products table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                product_id TEXT PRIMARY KEY,
                satisfiability_score INTEGER,
                image_authenticity_score INTEGER
            )
        ''')

        # Create ground_truth table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ground_truth (
                product_id TEXT PRIMARY KEY,
                satisfiability_score INTEGER,
                image_authenticity_score INTEGER,
                true_flagging_score FLOAT,
                true_status TEXT
            )
        ''')

        # Clear existing data
        cursor.execute("DELETE FROM products")
        cursor.execute("DELETE FROM ground_truth")

        # Generate records
        products_records = []
        ground_truth_records = []
        for _ in range(n_records):
            product_id = str(uuid.uuid4())
            satisfiability = random.randint(0, 99)
            image_authenticity = random.randint(0, 99)

            # Synthetic ground truth with more realistic variation
            true_score = (satisfiability * 0.5 + image_authenticity * 0.5) + random.uniform(-10, 10)
            true_score = np.clip(true_score, 0, 100)
            if true_score < 40:
                true_status = "Flagged"
            elif true_score < 60:
                true_status = "Sent for Human Review"
            else:
                true_status = "Genuine Product"

            products_records.append((product_id, satisfiability, image_authenticity))
            ground_truth_records.append((product_id, satisfiability, image_authenticity, true_score, true_status))

        # Insert into products table
        cursor.executemany('''
            INSERT INTO products (product_id, satisfiability_score, image_authenticity_score)
            VALUES (?, ?, ?)
        ''', products_records)

        # Insert into ground_truth table
        cursor.executemany('''
            INSERT INTO ground_truth (product_id, satisfiability_score, image_authenticity_score, true_flagging_score, true_status)
            VALUES (?, ?, ?, ?, ?)
        ''', ground_truth_records)

        # Commit changes and close connection
        conn.commit()
        print(f"Database created with {n_records} records in {db_name} (products and ground_truth tables)")
    except sqlite3.Error as e:
        print(f"Error creating database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    # Create database
    create_product_database(10000)
    # Calculate flagging scores
    df_predictions, test_indices = calculate_flagging_score()
    if df_predictions is not None:
        # Evaluate accuracy
        evaluate_model_accuracy(test_indices=test_indices)