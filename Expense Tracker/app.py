from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import sqlite3
import pandas as pd
import numpy as np
import os
import json
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import csv
import io
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from io import TextIOWrapper
import re

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import pickle
import joblib
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LinearRegression

now = datetime.now(timezone.utc).isoformat()

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.environ.get('SECRET_KEY', 'dev_secret')

# Database setup
def get_db_connection():
    conn = sqlite3.connect('finance_tracker.db')
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create users table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    # Create transactions table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('Income', 'Expense')),
            category TEXT NOT NULL,
            subcategory TEXT,
            note TEXT,
            amount REAL NOT NULL,
            mode TEXT DEFAULT 'manual',
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create category_thresholds table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS category_thresholds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            category TEXT NOT NULL,
            base_threshold REAL NOT NULL,
            dynamic_threshold REAL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            avg_monthly_spend REAL,
            std_monthly_spend REAL,
            FOREIGN KEY (user_id) REFERENCES users (id),
            UNIQUE(user_id, category)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# ML Model paths
MODEL_DIR = 'ml_models'
os.makedirs(MODEL_DIR, exist_ok=True)
CATEGORY_MODEL_PATH = os.path.join(MODEL_DIR, 'category_classifier.pkl')
AMOUNT_MODEL_PATH = os.path.join(MODEL_DIR, 'amount_regressor.pkl')
ENCODERS_PATH = os.path.join(MODEL_DIR, 'encoders.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')

# ML Models and encoders (initialized as None, will be loaded or trained as needed)
category_classifier = None
amount_regressor = None
label_encoders = None
tfidf_vectorizer = None

def load_ml_models():
    """Load ML models if they exist, otherwise return None"""
    global category_classifier, amount_regressor, label_encoders, tfidf_vectorizer
    
    try:
        if os.path.exists(CATEGORY_MODEL_PATH):
            category_classifier = joblib.load(CATEGORY_MODEL_PATH)
        
        if os.path.exists(AMOUNT_MODEL_PATH):
            amount_regressor = joblib.load(AMOUNT_MODEL_PATH)
            
        if os.path.exists(ENCODERS_PATH):
            label_encoders = joblib.load(ENCODERS_PATH)
            
        if os.path.exists(VECTORIZER_PATH):
            tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
    
    except Exception as e:
        print(f"Error loading ML models: {e}")
        category_classifier = None
        amount_regressor = None
        label_encoders = None
        tfidf_vectorizer = None

# Load models at startup
load_ml_models()

def train_ml_models(user_id=None):
    """Train ML models on transaction data with enhanced features"""
    global category_classifier, amount_regressor, label_encoders, tfidf_vectorizer
    
    conn = get_db_connection()
    
    # Get transaction data with more features
    if user_id:
        query = """
            SELECT 
                t.*,
                strftime('%Y-%m', date) as month,
                strftime('%w', date) as day_of_week,
                strftime('%H', date) as hour_of_day
            FROM transactions t 
            WHERE user_id = ? AND type = 'Expense'
        """
        params = (user_id,)
    else:
        query = """
            SELECT 
                t.*,
                strftime('%Y-%m', date) as month,
                strftime('%w', date) as day_of_week,
                strftime('%H', date) as hour_of_day
            FROM transactions t 
            WHERE type = 'Expense'
        """
        params = ()
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if len(df) < 10:
        return False, "Not enough transaction data to train models (need at least 10 expense records)"
    
    # Enhanced preprocessing
    df = df.dropna(subset=['category', 'amount'])
    df['note'] = df['note'].fillna('')
    df['subcategory'] = df['subcategory'].fillna('')
    df['mode'] = df['mode'].fillna('manual')
    
    # Add temporal features
    df['month'] = pd.to_datetime(df['month'] + '-01').dt.month
    df['day_of_week'] = df['day_of_week'].astype(int)
    df['hour_of_day'] = df['hour_of_day'].astype(int)
    
    # Initialize encoders
    le_mode = LabelEncoder()
    le_subcat = LabelEncoder()
    
    # Encode categorical features
    df['mode_encoded'] = le_mode.fit_transform(df['mode'])
    df['subcategory_encoded'] = le_subcat.fit_transform(df['subcategory'])
    
    # Store encoders
    label_encoders = {
        'mode': le_mode,
        'subcategory': le_subcat
    }
    
    # Enhanced text processing
    tfidf_vectorizer = TfidfVectorizer(
        max_features=100,
        ngram_range=(1, 2),
        stop_words='english'
    )
    X_note = tfidf_vectorizer.fit_transform(df['note']).toarray()
    
    # Combine all features
    X = np.concatenate([
        X_note,
        df[['amount', 'mode_encoded', 'subcategory_encoded', 
            'month', 'day_of_week', 'hour_of_day']].values
    ], axis=1)
    
    # Target variables
    y_category = df['category']
    y_amount = df['amount']
    
    # Train enhanced models
    category_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    category_classifier.fit(X, y_category)
    
    amount_regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    amount_regressor.fit(X, y_amount)
    
    # Save models and preprocessing tools
    joblib.dump(category_classifier, CATEGORY_MODEL_PATH)
    joblib.dump(amount_regressor, AMOUNT_MODEL_PATH)
    joblib.dump(label_encoders, ENCODERS_PATH)
    joblib.dump(tfidf_vectorizer, VECTORIZER_PATH)
    
    return True, "ML models trained successfully"

def predict_category_and_amount(note, subcategory='', mode='manual', amount=0):
    """Predict category and amount based on transaction details"""
    if not category_classifier or not amount_regressor:
        return None, None
    
    try:
        # Encode categorical features
        mode_encoded = label_encoders['mode'].transform([mode])[0]
        
        # Handle unseen subcategories by defaulting to most frequent
        try:
            subcategory_encoded = label_encoders['subcategory'].transform([subcategory])[0]
        except ValueError:
            # If subcategory is unseen, use the most frequent index
            subcategory_encoded = 0
        
        # Vectorize note
        note_features = tfidf_vectorizer.transform([note]).toarray()
        
        # Combine features
        features = np.concatenate([
            note_features,
            np.array([[amount, mode_encoded, subcategory_encoded]])
        ], axis=1)
        
        # Make predictions
        category = category_classifier.predict(features)[0]
        
        # For amount prediction, use a different model
        if amount == 0:  # Only predict amount if not provided
            predicted_amount = amount_regressor.predict(features)[0]
        else:
            predicted_amount = amount
        
        return category, predicted_amount
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None

def calculate_category_thresholds(user_id):
    """Calculate and update category thresholds based on historical spending"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get monthly spending by category for the last 6 months
        cursor.execute('''
            WITH monthly_spending AS (
                SELECT 
                    category,
                    strftime('%Y-%m', date) as month,
                    SUM(amount) as monthly_amount
                FROM transactions
                WHERE user_id = ? 
                    AND type = 'Expense'
                    AND date >= date('now', '-6 months')
                GROUP BY category, strftime('%Y-%m', date)
            )
            SELECT 
                category,
                AVG(monthly_amount) as avg_spend,
                SQRT(AVG((monthly_amount - AVG(monthly_amount)) * (monthly_amount - AVG(monthly_amount)))) as std_spend,
                MAX(monthly_amount) as max_spend
            FROM monthly_spending
            GROUP BY category
        ''', (user_id,))
        
        category_stats = cursor.fetchall()
        
        # Get user's monthly income
        cursor.execute('''
            SELECT AVG(monthly_income) as avg_income
            FROM (
                SELECT strftime('%Y-%m', date) as month,
                       SUM(amount) as monthly_income
                FROM transactions
                WHERE user_id = ? AND type = 'Income'
                GROUP BY strftime('%Y-%m', date)
            )
        ''', (user_id,))
        
        avg_monthly_income = cursor.fetchone()[0] or 0
        
        # Update thresholds for each category
        for stats in category_stats:
            category = stats['category']
            avg_spend = stats['avg_spend']
            std_spend = stats['std_spend']
            max_spend = stats['max_spend']
            
            # Calculate base threshold using the 50-30-20 rule and category type
            base_threshold = avg_monthly_income * get_category_allocation_percentage(category)
            
            # Calculate dynamic threshold using statistical analysis
            dynamic_threshold = avg_spend + (1.5 * std_spend)  # Using 1.5 standard deviations
            
            # Insert or update category thresholds
            cursor.execute('''
                INSERT INTO category_thresholds 
                    (user_id, category, base_threshold, dynamic_threshold, avg_monthly_spend, std_monthly_spend)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, category) DO UPDATE SET
                    base_threshold = excluded.base_threshold,
                    dynamic_threshold = excluded.dynamic_threshold,
                    avg_monthly_spend = excluded.avg_monthly_spend,
                    std_monthly_spend = excluded.std_monthly_spend,
                    last_updated = CURRENT_TIMESTAMP
            ''', (user_id, category, base_threshold, dynamic_threshold, avg_spend, std_spend))
        
        conn.commit()
        return True
        
    except Exception as e:
        print(f"Error calculating thresholds: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def get_category_allocation_percentage(category):
    """Get recommended allocation percentage based on category type"""
    # Define category types and their recommended allocations
    CATEGORY_ALLOCATIONS = {
        'Housing': 0.30,  # 30% of income
        'Transportation': 0.15,
        'Food': 0.15,
        'Utilities': 0.10,
        'Healthcare': 0.10,
        'Entertainment': 0.05,
        'Shopping': 0.05,
        'Education': 0.10,
        'Savings': 0.20,
        'Other': 0.05
    }
    
    # Default to 5% if category not found
    return CATEGORY_ALLOCATIONS.get(category, 0.05)

def generate_budget_advice(user_id):
    """Generate personalized budget advice using transaction data and ML insights"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get total income and expenses
    income = cursor.execute(
        'SELECT SUM(amount) FROM transactions WHERE user_id = ? AND type = "Income"',
        (user_id,)
    ).fetchone()[0] or 0
    
    expenses = cursor.execute(
        'SELECT SUM(amount) FROM transactions WHERE user_id = ? AND type = "Expense"',
        (user_id,)
    ).fetchone()[0] or 0
    
    # Get top spending categories with their totals
    top_categories = cursor.execute('''
        SELECT category, SUM(amount) as total
        FROM transactions
        WHERE user_id = ? AND type = 'Expense'
        GROUP BY category
        ORDER BY total DESC
        LIMIT 3
    ''', (user_id,)).fetchall()
    
    # Get spending trend (last 30 days vs. previous 30 days)
    current_month = cursor.execute('''
        SELECT SUM(amount) FROM transactions 
        WHERE user_id = ? AND type = 'Expense' 
        AND date >= date('now', '-30 days')
    ''', (user_id,)).fetchone()[0] or 0
    
    previous_month = cursor.execute('''
        SELECT SUM(amount) FROM transactions 
        WHERE user_id = ? AND type = 'Expense' 
        AND date < date('now', '-30 days') 
        AND date >= date('now', '-60 days')
    ''', (user_id,)).fetchone()[0] or 0
    
    conn.close()
    
    # Calculate net savings and savings rate
    net_savings = income - expenses
    savings_rate = (net_savings / income * 100) if income > 0 else 0
    
    # Generate advice
    advice = []
    
    # Basic financial summary
    advice.append(f"💰 Total Income: ₹{income:.2f}")
    advice.append(f"💸 Total Expenses: ₹{expenses:.2f}")
    advice.append(f"🏦 Net Savings: ₹{net_savings:.2f} ({savings_rate:.1f}% of income)")
    
    # Top spending categories
    if top_categories:
        advice.append("\n📊 Top spending categories:")
        for i, (category, amount) in enumerate(top_categories):
            advice.append(f"  {i+1}. {category}: ₹{amount:.2f}")
        
        # Recommendation for highest category
        highest_category = top_categories[0][0]
        advice.append(f"\n👉 Recommendation: Consider reducing spending in '{highest_category}' as it's your highest expense.")
    
    # Spending trend
    if current_month > 0 and previous_month > 0:
        percent_change = ((current_month - previous_month) / previous_month) * 100
        if percent_change > 10:
            advice.append(f"\n⚠️ Your spending increased by {percent_change:.1f}% compared to the previous month.")
        elif percent_change < -10:
            advice.append(f"\n✅ Great job! Your spending decreased by {abs(percent_change):.1f}% compared to the previous month.")
    
    # 50/30/20 rule check (50% needs, 30% wants, 20% savings)
    if income > 0:
        if savings_rate < 20:
            advice.append("\n💡 Try to aim for at least 20% savings rate for good financial health.")
        else:
            advice.append("\n🌟 Your savings rate is excellent! Keep up the good work.")
    
    return "\n".join(advice)

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('dashboard.html')

@app.route('/manual_entry')
def manual_entry():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('manual_entry.html')

@app.route('/csv_upload')
def csv_upload():
    if 'user_id' not in session:
        return redirect(url_for('index'))
    return render_template('csv_upload.html')

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    
    # Validate data
    if not data or not data.get('username') or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Missing required fields'}), 400
    
    username = data.get('username')
    email = data.get('email')
    password = generate_password_hash(data.get('password'))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', 
                      (username, email, password))
        conn.commit()
        return jsonify({'message': 'Signup successful!'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username or email already exists'}), 400
    finally:
        conn.close()

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    
    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Basic password strength check
    if len(data.get('password')) < 6:
        return jsonify({'error': 'Password must be at least 6 characters long'}), 400

    
    email = data.get('email')
    password = data.get('password')
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    user = cursor.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
    conn.close()
    
    if user and check_password_hash(user['password'], password):
        session['user_id'] = user['id']
        session['username'] = user['username']
        return jsonify({'message': 'Login successful!'}), 200
    
    return jsonify({'error': 'Invalid email or password'}), 401

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/user_info')
def user_info():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    return jsonify({'username': session.get('username')})

@app.route('/add_transaction', methods=['POST'])
def add_transaction():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    app.logger.info(f"[add_transaction] user_id={session['user_id']} Received transaction data: {request.form}")
    print("request.form:  ",request.form)

    try:
        # Get and validate required fields
        txn_type = request.form.get('type')
        print("type: ",txn_type)
        amount = request.form.get('amount')
        print("amount:",amount)
        note = request.form.get('note', '')
        mode = request.form.get('mode', 'manual')

        app.logger.info(f"[add_transaction] Processing transaction - Type: {txn_type}, Amount: {amount}")

        if not txn_type:
            app.logger.error("Transaction type is missing")
            return jsonify({'error': 'Transaction type is required'}), 400

        if not amount:
            app.logger.error("Amount is missing")
            return jsonify({'error': 'Amount is required'}), 400

        try:
            amount = float(str(amount).replace(',', '').replace('₹', '').strip())
            if amount <= 0:
                app.logger.error(f"Invalid amount value: {amount}")
                return jsonify({'error': 'Amount must be greater than 0'}), 400
        except ValueError:
            app.logger.error(f"Invalid amount value: {amount}")
            return jsonify({'error': 'Invalid amount format'}), 400

        # Handle transaction type and category
        if txn_type == 'Income':
            category = 'Income'
            subcategory = ''
        else:
            category = request.form.get('category')
            subcategory = request.form.get('subcategory', '')
            if not category:
                app.logger.error("Category missing for expense transaction")
                return jsonify({'error': 'Category is required for expenses'}), 400

        # Get current timestamp
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            # Insert the transaction
            cursor.execute('''
                INSERT INTO transactions (
                    user_id, type, category, subcategory, 
                    note, amount, mode, date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session['user_id'], txn_type, category, subcategory,
                note, amount, mode, current_time
            ))
            conn.commit()

            # Get the inserted transaction
            cursor.execute('''
                SELECT * FROM transactions 
                WHERE id = last_insert_rowid()
            ''')
            new_transaction = cursor.fetchone()
            app.logger.info(f"[add_transaction] New transaction added: {dict(new_transaction)}")

            # Get updated totals with explicit type checks
            cursor.execute('''
                SELECT COALESCE(SUM(amount), 0) as total_income
                FROM transactions 
                WHERE user_id = ? AND type = 'Income'
            ''', (session['user_id'],))
            income_result = cursor.fetchone()
            income_total = float(income_result['total_income']) if income_result and income_result['total_income'] is not None else 0

            cursor.execute('''
                SELECT COALESCE(SUM(amount), 0) as total_expenses
                FROM transactions 
                WHERE user_id = ? AND type = 'Expense'
            ''', (session['user_id'],))
            expense_result = cursor.fetchone()
            expense_total = float(expense_result['total_expenses']) if expense_result and expense_result['total_expenses'] is not None else 0

            net_savings = income_total - expense_total

            app.logger.info(f"[add_transaction] Updated totals - Income: {income_total}, Expenses: {expense_total}, Savings: {net_savings}")

            return jsonify({
                'message': 'Transaction added successfully',
                'transaction': {
                    'id': new_transaction['id'],
                    'type': new_transaction['type'],
                    'category': new_transaction['category'],
                    'subcategory': new_transaction['subcategory'],
                    'note': new_transaction['note'],
                    'amount': float(new_transaction['amount']),
                    'mode': new_transaction['mode'],
                    'date': new_transaction['date']
                },
                'totals': {
                    'income': income_total,
                    'expenses': expense_total,
                    'savings': net_savings
                }
            })

        except sqlite3.Error as e:
            app.logger.error(f"Database error in add_transaction: {str(e)}")
            conn.rollback()
            return jsonify({'error': 'Database error occurred'}), 500

        finally:
            conn.close()

    except Exception as e:
        app.logger.error(f"Unexpected error in add_transaction: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    conn = None
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Not logged in'}), 401
            
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'}), 400
        
        # Read the CSV file
        stream = TextIOWrapper(file.stream, encoding='utf-8-sig')
        csv_reader = csv.DictReader(stream)
        
        # Validate required columns
        required_columns = {'category', 'amount', 'type'}
        if not all(col in csv_reader.fieldnames for col in required_columns):
            missing = required_columns - set(csv_reader.fieldnames)
            return jsonify({'error': f'CSV missing required columns: {", ".join(missing)}'}), 400
        
        # Store CSV data in memory
        csv_data = list(csv_reader)
        if not csv_data:
            return jsonify({'error': 'CSV file is empty'}), 400
        
        # Initialize database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Begin transaction
        conn.execute('BEGIN TRANSACTION')
        
        # Clear existing transactions for this user
        cursor.execute('DELETE FROM transactions WHERE user_id = ?', (session['user_id'],))
        
        # Process transactions
        imported_transactions = []
        transactions_by_category = defaultdict(list)
        row_count = 0
        errors = []
        income_total = 0
        expense_total = 0
        
        for row_num, row in enumerate(csv_data, 1):
            try:
                # Validate and clean data
                category = row['category'].strip()
                if not category:
                    errors.append(f"Row {row_num}: Empty category")
                    continue
                
                try:
                    amount = float(str(row['amount']).replace(',', '').replace('₹', '').strip())
                except (ValueError, TypeError):
                    errors.append(f"Row {row_num}: Invalid amount '{row['amount']}'")
                    continue
                
                # Get optional fields with defaults
                description = row.get('description', '').strip()
                date_str = row.get('date', datetime.now().strftime('%Y-%m-%d'))
                
                # Handle transaction type
                transaction_type = row.get('type', 'Expense').strip().title()
                if transaction_type not in ['Income', 'Expense']:
                    transaction_type = 'Expense'
                
                # Ensure amount is positive
                amount = abs(amount)
                
                # Update totals
                if transaction_type == 'Income':
                    income_total += amount
                else:
                    expense_total += amount
                
                # Insert transaction
                cursor.execute('''
                    INSERT INTO transactions (user_id, type, category, amount, note, date)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (session['user_id'], transaction_type, category, amount, description, date_str))
                
                transaction_data = {
                    'type': transaction_type,
                    'category': category,
                    'amount': amount,
                    'description': description,
                    'date': date_str
                }
                
                # Store transaction by category
                category_key = f"{transaction_type}_{category}"
                transactions_by_category[category_key].append(transaction_data)
                imported_transactions.append(transaction_data)
                row_count += 1
                
            except Exception as e:
                errors.append(f"Row {row_num}: {str(e)}")
                continue
        
        if row_count == 0:
            conn.rollback()
            error_msg = "No valid transactions found in the CSV file"
            if errors:
                error_msg += f". Errors: {'; '.join(errors)}"
            return jsonify({'error': error_msg}), 400
        
        # Calculate net savings
        net_savings = income_total - expense_total
        savings_rate = (net_savings / income_total * 100) if income_total > 0 else 0
        
        # Get category-wise breakdown with top 3 transactions
        category_transactions = {}
        cursor.execute('''
            SELECT type, category, COUNT(*) as count, SUM(amount) as total
            FROM transactions 
            WHERE user_id = ?
            GROUP BY type, category
        ''', (session['user_id'],))
        
        categories = cursor.fetchall()
        
        for cat in categories:
            key = f"{cat['type']}_{cat['category']}"
            # Get all transactions for this category
            category_txns = transactions_by_category[key]
            
            # Sort by amount and get top 3
            top_transactions = sorted(
                category_txns,
                key=lambda x: x['amount'],
                reverse=True
            )[:3]
            
            category_transactions[key] = {
                'type': cat['type'],
                'category': cat['category'],
                'count': cat['count'],
                'total': float(cat['total']),
                'top_transactions': top_transactions
            }
        
        # Commit the transaction
        conn.commit()
        
        response = {
            'message': f'Successfully imported {row_count} transactions',
            'summary': {
                'total_income': income_total,
                'total_expenses': expense_total,
                'net_savings': net_savings,
                'savings_rate': savings_rate
            },
            'category_breakdown': category_transactions
        }
        
        if errors:
            response['warnings'] = errors
            
        return jsonify(response)
        
    except Exception as e:
        if conn:
            conn.rollback()
        app.logger.error(f"Error processing CSV: {str(e)}")
        return jsonify({'error': f'Error processing CSV file: {str(e)}'}), 500
        
    finally:
        if conn:
            conn.close()

@app.route('/clear_transactions', methods=['POST'])
def clear_transactions():
    """Clear all transactions for the current user"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM transactions WHERE user_id = ?', (session['user_id'],))
        conn.commit()
        return jsonify({'message': 'All transactions cleared successfully'})
    except Exception as e:
        return jsonify({'error': 'Failed to clear transactions'}), 500

@app.route('/get_transactions')
def get_transactions():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get filter parameters
        transaction_type = request.args.get('type', 'all')
        date_filter = request.args.get('date_filter', 'all')
        category = request.args.get('category', 'all')
        
        # Base query
        query = '''
            SELECT * FROM transactions 
            WHERE user_id = ?
        '''
        params = [session['user_id']]
        
        # Add type filter
        if transaction_type != 'all':
            query += ' AND type = ?'
            params.append(transaction_type)
            
        # Add category filter
        if category != 'all':
            query += ' AND category = ?'
            params.append(category)
        
        # Add date filter
        if date_filter == 'today':
            query += ' AND date >= date("now", "start of day")'
        elif date_filter == 'this_month':
            query += ' AND date >= date("now", "start of month")'
        
        # Add ordering
        query += ' ORDER BY date DESC'
        
        # Get transactions
        cursor.execute(query, params)
        transactions = cursor.fetchall()
        
        # Get totals with the same filters
        total_query = '''
            SELECT type, SUM(amount) as total
            FROM transactions 
            WHERE user_id = ?
        '''
        total_params = [session['user_id']]
        
        if transaction_type != 'all':
            total_query += ' AND type = ?'
            total_params.append(transaction_type)
            
        if category != 'all':
            total_query += ' AND category = ?'
            total_params.append(category)
            
        if date_filter == 'today':
            total_query += ' AND date >= date("now", "start of day")'
        elif date_filter == 'this_month':
            total_query += ' AND date >= date("now", "start of month")'
            
        total_query += ' GROUP BY type'
        
        cursor.execute(total_query, total_params)
        totals = cursor.fetchall()
        
        income_total = 0
        expense_total = 0
        for total in totals:
            if total['type'] == 'Income':
                income_total = float(total['total'])
            else:
                expense_total = float(total['total'])
        
        net_savings = income_total - expense_total
        
        # Get category breakdown
        category_query = '''
            SELECT type, category, COUNT(*) as count, SUM(amount) as total
            FROM transactions 
            WHERE user_id = ?
        '''
        category_params = [session['user_id']]
        
        if transaction_type != 'all':
            category_query += ' AND type = ?'
            category_params.append(transaction_type)
            
        if category != 'all':
            category_query += ' AND category = ?'
            category_params.append(category)
            
        if date_filter == 'today':
            category_query += ' AND date >= date("now", "start of day")'
        elif date_filter == 'this_month':
            category_query += ' AND date >= date("now", "start of month")'
            
        category_query += ' GROUP BY type, category'
        
        cursor.execute(category_query, category_params)
        categories = cursor.fetchall()
        
        # Prepare response data
        transaction_list = []
        for txn in transactions:
            transaction_list.append({
                'id': txn['id'],
                'type': txn['type'],
                'category': txn['category'],
                'subcategory': txn['subcategory'],
                'note': txn['note'],
                'amount': float(txn['amount']),
                'mode': txn['mode'],
                'date': txn['date']
            })
        
        category_breakdown = {}
        for cat in categories:
            key = f"{cat['type']}_{cat['category']}"
            
            # Get top 3 transactions for this category
            cursor.execute('''
                SELECT * FROM transactions 
                WHERE user_id = ? AND type = ? AND category = ?
                ORDER BY amount DESC LIMIT 3
            ''', (session['user_id'], cat['type'], cat['category']))
            
            top_transactions = []
            for t in cursor.fetchall():
                top_transactions.append({
                    'amount': float(t['amount']),
                    'date': t['date'],
                    'description': t['note']
                })
            
            category_breakdown[key] = {
                'type': cat['type'],
                'category': cat['category'],
                'count': cat['count'],
                'total': float(cat['total']),
                'top_transactions': top_transactions
            }
        
        return jsonify({
            'transactions': transaction_list,
            'summary': {
                'total_income': income_total,
                'total_expenses': expense_total,
                'net_savings': net_savings
            },
            'category_breakdown': category_breakdown
        })
        
    except Exception as e:
        app.logger.error(f"[get_transactions] Error: {str(e)}")
        return jsonify({'error': 'An error occurred while fetching transactions'}), 500
    finally:
        if 'conn' in locals():
            conn.close()

@app.route('/get_budget_advice')
def get_budget_advice():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get current month's data
        cursor.execute('''
            SELECT type, category, SUM(amount) as total
            FROM transactions 
            WHERE user_id = ? 
            AND date >= date('now', 'start of month')
            GROUP BY type, category
        ''', (session['user_id'],))
        current_month = cursor.fetchall()

        # Get previous month's data for comparison
        cursor.execute('''
            SELECT type, category, SUM(amount) as total
            FROM transactions 
            WHERE user_id = ? 
            AND date >= date('now', 'start of month', '-1 month')
            AND date < date('now', 'start of month')
            GROUP BY type, category
        ''', (session['user_id'],))
        previous_month = cursor.fetchall()

        # Calculate total income and expenses
        income = sum(float(row['total']) for row in current_month if row['type'] == 'Income')
        expenses = sum(float(row['total']) for row in current_month if row['type'] == 'Expense')

        # Define category thresholds (percentage of income)
        CATEGORY_THRESHOLDS = {
            'Housing': 0.30,      # 30% of income
            'Food': 0.15,         # 15% of income
            'Transportation': 0.15,# 15% of income
            'Utilities': 0.10,    # 10% of income
            'Healthcare': 0.10,   # 10% of income
            'Entertainment': 0.05, # 5% of income
            'Shopping': 0.05,     # 5% of income
            'Education': 0.10,    # 10% of income
            'Savings': 0.20,
            'Other': 0.05
        }

        advice = []
        urgent_alerts = []
        warnings = []
        positive_notes = []

        # Calculate net savings
        net_savings = income - expenses
        savings_rate = (net_savings / income * 100) if income > 0 else 0

        # Add savings advice if needed
        if income > 0:
            if savings_rate < 0:
                urgent_alerts.append("🚨 ALERT: You are spending more than your income this month!")
            elif savings_rate < 20:
                warnings.append(f"⚠️ Your savings rate is only {savings_rate:.1f}%. Aim for at least 20% savings.")
            elif savings_rate > 30:
                positive_notes.append(f"✅ Great job! Your savings rate is {savings_rate:.1f}%.")

        # Analyze each expense category
        for row in current_month:
            if row['type'] == 'Expense':
                category = row['category']
                amount = float(row['total'])
                
                # Get threshold for category
                threshold_percentage = CATEGORY_THRESHOLDS.get(category, 0.05)
                threshold_amount = income * threshold_percentage

                # Compare with threshold
                if amount > threshold_amount:
                    percentage_over = ((amount - threshold_amount) / threshold_amount * 100)
                    if percentage_over > 50:
                        urgent_alerts.append(f"🚨 {category}: Spending is {percentage_over:.1f}% over recommended limit!")
                    elif percentage_over > 20:
                        warnings.append(f"⚠️ {category}: Spending is {percentage_over:.1f}% over recommended limit.")
                elif amount < threshold_amount * 0.5:
                    positive_notes.append(f"✅ {category}: Good control on spending (under budget).")

                # Compare with previous month
                prev_amount = next((float(row['total']) for row in previous_month 
                                 if row['type'] == 'Expense' and row['category'] == category), 0)
                if prev_amount > 0:
                    change = ((amount - prev_amount) / prev_amount * 100)
                    if change > 30:
                        warnings.append(f"📈 {category}: Spending increased by {change:.1f}% from last month.")
                    elif change < -20:
                        positive_notes.append(f"📉 {category}: Reduced spending by {abs(change):.1f}% from last month.")

        # Build final advice message
        if urgent_alerts:
            advice.extend(["🔴 Urgent Actions Needed:", *urgent_alerts, ""])

        if warnings:
            advice.extend(["🟡 Warnings:", *warnings, ""])

        if positive_notes:
            advice.extend(["🟢 Positive Notes:", *positive_notes, ""])

        # Add general recommendations
        if income > 0:
            advice.append("\n💡 Recommendations:")
            if savings_rate < 20:
                advice.append("• Consider creating a monthly budget for each category")
                advice.append("• Look for ways to reduce non-essential expenses")
                advice.append("• Track your daily expenses more closely")
            elif expenses > income * 0.8:
                advice.append("• Try to maintain or reduce current spending levels")
                advice.append("• Consider saving more for emergencies")
            else:
                advice.append("• Consider investing your savings for better returns")
                advice.append("• Keep maintaining your good financial habits")

        return jsonify({'advice': "\n".join(advice) if advice else "Add more transactions to get personalized advice."})

    except Exception as e:
        app.logger.error(f"Error generating budget advice: {str(e)}")
        return jsonify({'error': 'Unable to generate advice at this time'}), 500

    finally:
        if 'conn' in locals():
            conn.close()

@app.route('/predict_category', methods=['POST'])
def predict_category():
    """API endpoint to predict category based on transaction details"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    # Check if models are loaded
    if not category_classifier or not tfidf_vectorizer or not label_encoders:
        # Try to train models first
        success, message = train_ml_models(session['user_id'])
        if not success:
            return jsonify({'error': message}), 400
    
    # Get input data
    note = request.form.get('note', '')
    subcategory = request.form.get('subcategory', '')
    mode = request.form.get('mode', 'manual')
    
    try:
        amount = float(request.form.get('amount', 0))
    except (ValueError, TypeError):
        amount = 0
    
    # Make prediction
    predicted_category, predicted_amount = predict_category_and_amount(
        note, subcategory, mode, amount
    )
    
    if predicted_category is None:
        return jsonify({'error': 'Unable to make prediction'}), 400
    
    return jsonify({
        'predicted_category': predicted_category,
        'predicted_amount': float(predicted_amount) if amount == 0 else None
    })

@app.route('/train_models', methods=['POST'])
def train_models_endpoint():
    """API endpoint to manually trigger model training"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    success, message = train_ml_models(session['user_id'])
    
    if success:
        return jsonify({'message': message})
    else:
        return jsonify({'error': message}), 400

@app.route('/get_spending_insights')
def get_spending_insights():
    """Get enhanced ML-powered insights about spending patterns"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    conn = get_db_connection()
    
    # Get detailed transaction data
    df = pd.read_sql_query('''
        SELECT 
            strftime('%Y-%m-%d', date) as date,
            strftime('%w', date) as day_of_week,
            strftime('%H', date) as hour_of_day,
            category,
            amount,
            note,
            type,
            subcategory
        FROM transactions
        WHERE user_id = ?
        ORDER BY date DESC
    ''', conn, params=(session['user_id'],))
    
    conn.close()
    
    if len(df) == 0:
        return jsonify({
            'message': 'Not enough data for insights',
            'insights': []
        })
    
    insights = []
    
    # Basic financial summary
    total_expenses = df[df['type'] == 'Expense']['amount'].sum()
    total_income = df[df['type'] == 'Income']['amount'].sum()
    net_savings = total_income - total_expenses
    savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0
    
    insights.append({
        'type': 'summary',
        'title': 'Financial Summary',
        'details': [
            f"💰 Total Income: ₹{total_income:,.2f}",
            f"💸 Total Expenses: ₹{total_expenses:,.2f}",
            f"🏦 Net Savings: ₹{net_savings:,.2f} ({savings_rate:.1f}% of income)"
        ]
    })
    
    # Income Analysis
    if total_income > 0:
        income_df = df[df['type'] == 'Income']
        income_categories = income_df.groupby('category')['amount'].agg(['sum', 'count']).reset_index()
        income_categories['percentage'] = (income_categories['sum'] / total_income * 100).round(1)
        
        income_insights = []
        for _, row in income_categories.iterrows():
            income_insights.append(
                f"• {row['category']}: ₹{row['sum']:,.2f} ({row['percentage']}% of total income)"
            )
        
        if income_insights:
            insights.append({
                'type': 'income_analysis',
                'title': 'Income Analysis',
                'details': income_insights
            })
    
    # Expense Analysis
    expense_df = df[df['type'] == 'Expense']
    expense_categories = expense_df.groupby('category')['amount'].agg(['sum', 'count']).reset_index()
    expense_categories['percentage'] = (expense_categories['sum'] / total_expenses * 100).round(1)
    
    expense_insights = []
    for _, row in expense_categories.iterrows():
        expense_insights.append(
            f"• {row['category']}: ₹{row['sum']:,.2f} ({row['percentage']}% of total expenses)"
        )
    
    if expense_insights:
        insights.append({
            'type': 'expense_analysis',
            'title': 'Expense Analysis',
            'details': expense_insights
        })
    
    # Savings Analysis
    if total_income > 0:
        savings_insights = []
        if savings_rate >= 50:
            savings_insights.append("🌟 Excellent! Your savings rate is above 50%.")
        elif savings_rate >= 30:
            savings_insights.append("✅ Good job! Your savings rate is above 30%.")
        elif savings_rate >= 20:
            savings_insights.append("👍 Your savings rate is healthy at 20%.")
        else:
            savings_insights.append("💡 Consider increasing your savings rate to at least 20% for better financial health.")
        
        insights.append({
            'type': 'savings_analysis',
            'title': 'Savings Analysis',
            'details': savings_insights
        })
    
    # Temporal Analysis
    temporal_patterns = analyze_temporal_patterns(expense_df)
    if temporal_patterns:
        insights.append({
            'type': 'temporal_analysis',
            'title': 'Time-based Spending Patterns',
            'details': [pattern['message'] for pattern in temporal_patterns]
        })
    
    # Category-specific Analysis
    category_insights = []
    for category in expense_df['category'].unique():
        cat_data = expense_df[expense_df['category'] == category]
        avg_amount = cat_data['amount'].mean()
        total_amount = cat_data['amount'].sum()
        percentage = (total_amount / total_expenses * 100) if total_expenses > 0 else 0
        
        if percentage > 30:
            category_insights.append(
                f"⚠️ {category} represents {percentage:.1f}% of your expenses. Consider reviewing spending in this category."
            )
    
    if category_insights:
        insights.append({
            'type': 'category_analysis',
            'title': 'Category-specific Insights',
            'details': category_insights
        })
    
    # Recommendations
    recommendations = []
    
    # Income-based recommendations
    if total_income > 0:
        if savings_rate < 20:
            recommendations.append({
                'priority': 'high',
                'message': "Your savings rate is below 20%. Consider reducing expenses or increasing income.",
                'action': "Review your largest expense categories and look for areas to cut back."
            })
        elif savings_rate > 50:
            recommendations.append({
                'priority': 'low',
                'message': "Excellent savings rate! Consider investing your savings for better returns.",
                'action': "Look into investment options that match your risk tolerance."
            })
    
    # Expense-based recommendations
    for category, row in expense_categories.iterrows():
        if row['percentage'] > 30:
            recommendations.append({
                'priority': 'high',
                'message': f"High spending in {category} category ({row['percentage']:.1f}% of expenses).",
                'action': f"Review your {category} expenses and look for ways to reduce costs."
            })
    
    if recommendations:
        insights.append({
            'type': 'recommendations',
            'title': 'Personalized Recommendations',
            'details': recommendations
        })
    
    return jsonify({
        'insights': insights
    })

def extract_merchant_info(note):
    """Extract merchant information from transaction notes"""
    # Common merchant patterns
    patterns = {
        'online': r'(amazon|flipkart|myntra|swiggy|zomato)',
        'retail': r'(store|shop|market|mall)',
        'food': r'(restaurant|cafe|food|dining)',
        'transport': r'(uber|ola|metro|bus|train)',
        'utility': r'(electricity|water|gas|internet)'
    }
    
    merchant_type = 'other'
    for type_name, pattern in patterns.items():
        if re.search(pattern, note.lower()):
            merchant_type = type_name
            break
    
    return {
        'merchant_type': merchant_type,
        'is_online': 1 if merchant_type == 'online' else 0
    }

def analyze_merchant_patterns(df):
    """Analyze spending patterns by merchant"""
    merchant_patterns = []
    
    # Group by merchant type
    merchant_stats = df.groupby('merchant_type').agg({
        'amount': ['count', 'mean', 'sum'],
        'date': 'nunique'
    }).round(2)
    
    for merchant_type, stats in merchant_stats.iterrows():
        total_transactions = stats[('amount', 'count')]
        avg_amount = stats[('amount', 'mean')]
        total_spent = stats[('amount', 'sum')]
        unique_days = stats[('date', 'nunique')]
        
        # Calculate frequency
        if unique_days > 0:
            frequency = total_transactions / unique_days
        else:
            frequency = 0
        
        pattern = {
            'merchant_type': merchant_type,
            'total_transactions': total_transactions,
            'avg_amount': avg_amount,
            'total_spent': total_spent,
            'frequency': frequency,
            'unique_days': unique_days
        }
        
        # Add risk assessment
        if frequency > 1.5:  # More than 1.5 transactions per day on average
            pattern['risk_level'] = 'high'
            pattern['message'] = f"Frequent spending at {merchant_type} merchants: {frequency:.1f} transactions per day"
        elif frequency > 0.5:  # More than 0.5 transactions per day on average
            pattern['risk_level'] = 'medium'
            pattern['message'] = f"Regular spending at {merchant_type} merchants: {frequency:.1f} transactions per day"
        else:
            pattern['risk_level'] = 'low'
            pattern['message'] = f"Occasional spending at {merchant_type} merchants: {frequency:.1f} transactions per day"
        
        merchant_patterns.append(pattern)
    
    return merchant_patterns

def analyze_temporal_patterns(df):
    """Analyze spending patterns by time"""
    temporal_patterns = []
    
    # Convert to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = pd.to_numeric(df['hour_of_day'])
    df['day_of_week'] = pd.to_numeric(df['day_of_week'])
    
    # Time of day analysis
    time_slots = {
        'morning': (6, 12),
        'afternoon': (12, 18),
        'evening': (18, 22),
        'night': (22, 6)
    }
    
    for slot_name, (start_hour, end_hour) in time_slots.items():
        if start_hour < end_hour:
            slot_data = df[(df['hour'] >= start_hour) & (df['hour'] < end_hour)]
        else:  # Handle overnight slot
            slot_data = df[(df['hour'] >= start_hour) | (df['hour'] < end_hour)]
        
        if len(slot_data) > 0:
            avg_amount = slot_data['amount'].mean()
            total_spent = slot_data['amount'].sum()
            transaction_count = len(slot_data)
            
            pattern = {
                'time_slot': slot_name,
                'avg_amount': avg_amount,
                'total_spent': total_spent,
                'transaction_count': transaction_count
            }
            
            # Add insights
            if avg_amount > df['amount'].mean() * 1.2:
                pattern['message'] = f"Higher than average spending during {slot_name}: ₹{avg_amount:,.2f} per transaction"
            elif avg_amount < df['amount'].mean() * 0.8:
                pattern['message'] = f"Lower than average spending during {slot_name}: ₹{avg_amount:,.2f} per transaction"
            else:
                pattern['message'] = f"Typical spending during {slot_name}: ₹{avg_amount:,.2f} per transaction"
            
            temporal_patterns.append(pattern)
    
    return temporal_patterns

def analyze_spending_patterns(df):
    """Analyze spending patterns using clustering"""
    patterns = []
    
    # Prepare features for clustering
    features = df.groupby('date').agg({
        'amount': ['sum', 'count', 'mean', 'std']
    }).fillna(0)
    
    features.columns = ['daily_total', 'transaction_count', 'avg_amount', 'amount_std']
    
    if len(features) < 5:  # Need enough data points for meaningful clustering
        return patterns
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=min(3, len(features)), random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Analyze each cluster
    for cluster_id in range(kmeans.n_clusters):
        cluster_days = features[clusters == cluster_id]
        if len(cluster_days) > 0:
            pattern = {
                'type': 'cluster',
                'cluster_id': cluster_id,
                'days_count': len(cluster_days),
                'avg_daily_spend': cluster_days['daily_total'].mean(),
                'avg_transactions': cluster_days['transaction_count'].mean(),
                'avg_amount': cluster_days['avg_amount'].mean()
            }
            
            # Determine pattern type
            if pattern['avg_transactions'] > features['transaction_count'].mean() * 1.2:
                pattern['pattern_type'] = 'frequent_small'
                pattern['message'] = f"Frequent small purchases pattern: {len(cluster_days)} days with average {pattern['avg_transactions']:.1f} transactions per day"
            elif pattern['avg_amount'] > features['avg_amount'].mean() * 1.5:
                pattern['pattern_type'] = 'occasional_large'
                pattern['message'] = f"Occasional large purchases pattern: {len(cluster_days)} days with average transaction of ₹{pattern['avg_amount']:,.2f}"
            else:
                pattern['pattern_type'] = 'regular'
                pattern['message'] = f"Regular spending pattern: {len(cluster_days)} days with average daily spend of ₹{pattern['avg_daily_spend']:,.2f}"
            
            patterns.append(pattern)
    
    return patterns

def train_category_models(df):
    """Train regression models for each spending category"""
    category_models = {}
    
    for category in df['category'].unique():
        cat_data = df[df['category'] == category].copy()
        if len(cat_data) < 10:  # Need enough data points
            continue
        
        # Prepare features
        cat_data['month'] = pd.to_datetime(cat_data['date']).dt.month
        cat_data['day_of_week'] = pd.to_numeric(cat_data['day_of_week'])
        cat_data['hour_of_day'] = pd.to_numeric(cat_data['hour_of_day'])
        
        X = cat_data[['month', 'day_of_week', 'hour_of_day']]
        y = cat_data['amount']
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        category_models[category] = {
            'model': model,
            'avg_amount': y.mean(),
            'std_amount': y.std()
        }
    
    return category_models

def predict_category_spending(category_models, category, features):
    """Predict expected spending for a category"""
    if category not in category_models:
        return None
    
    model_info = category_models[category]
    predicted = model_info['model'].predict([features])[0]
    
    return {
        'predicted': predicted,
        'avg_amount': model_info['avg_amount'],
        'std_amount': model_info['std_amount']
    }

@app.route('/get_category_advice')
def get_category_advice():
    """Get advice for a specific category"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    category = request.args.get('category')
    if not category:
        return jsonify({'error': 'Category not specified'}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Define category-specific thresholds and advice
        CATEGORY_THRESHOLDS = {
            'Housing': {
                'threshold': 40000,  # Higher threshold for housing
                'warning_threshold': 70,  # Warning at 70% of threshold
                'critical_threshold': 85,  # Critical at 85% of threshold
                'advice': [
                    "Consider negotiating rent or finding a more affordable place",
                    "Look for ways to reduce utility costs",
                    "Consider getting a roommate to split costs"
                ],
                'critical_advice': [
                    "Review your lease agreement for any cost-saving opportunities",
                    "Consider downsizing or relocating to a more affordable area",
                    "Evaluate utility usage and implement energy-saving measures"
                ]
            },
            'Transportation': {
                'threshold': 20000,  # Moderate threshold for transportation
                'warning_threshold': 75,  # Warning at 75% of threshold
                'critical_threshold': 90,  # Critical at 90% of threshold
                'advice': [
                    "Consider using public transportation more often",
                    "Look for carpooling opportunities",
                    "Plan your trips to save fuel"
                ],
                'critical_advice': [
                    "Consider switching to a more fuel-efficient vehicle",
                    "Look into monthly public transportation passes for cost savings",
                    "Explore work-from-home opportunities to reduce commuting costs"
                ]
            },
            'Food': {
                'threshold': 15000,  # Standard threshold for food
                'warning_threshold': 80,  # Warning at 80% of threshold
                'critical_threshold': 95,  # Critical at 95% of threshold
                'advice': [
                    "Try meal planning to reduce food waste",
                    "Cook more meals at home",
                    "Look for grocery deals and buy in bulk"
                ],
                'critical_advice': [
                    "Create a strict grocery budget and stick to it",
                    "Use apps to find the best deals and discounts",
                    "Consider growing some basic vegetables at home"
                ]
            },
            'Healthcare': {
                'threshold': 10000,  # Priority threshold for healthcare
                'warning_threshold': 85,  # Warning at 85% of threshold
                'critical_threshold': 95,  # Critical at 95% of threshold
                'advice': [
                    "Consider getting health insurance if you haven't",
                    "Look for preventive care options",
                    "Compare prices for medications"
                ],
                'critical_advice': [
                    "Review your health insurance plan for better coverage",
                    "Ask about generic medication alternatives",
                    "Look into preventive care to avoid costly treatments"
                ]
            },
            'Entertainment': {
                'threshold': 8000,  # Lower threshold for entertainment
                'warning_threshold': 65,  # Warning at 65% of threshold
                'critical_threshold': 80,  # Critical at 80% of threshold
                'advice': [
                    "Look for free or low-cost entertainment options",
                    "Use streaming services instead of multiple subscriptions",
                    "Take advantage of happy hours and deals"
                ],
                'critical_advice': [
                    "Cancel unused entertainment subscriptions",
                    "Look for free community events and activities",
                    "Consider sharing subscription costs with family or friends"
                ]
            },
            'Shopping': {
                'threshold': 12000,  # Moderate threshold for shopping
                'warning_threshold': 70,  # Warning at 70% of threshold
                'critical_threshold': 85,  # Critical at 85% of threshold
                'advice': [
                    "Make a shopping list and stick to it",
                    "Look for sales and discounts",
                    "Consider second-hand or refurbished items"
                ],
                'critical_advice': [
                    "Review all non-essential purchases",
                    "Implement a waiting period for large purchases",
                    "Use cashback and reward programs"
                ]
            },
            'Education': {
                'threshold': 25000,  # Higher threshold for education
                'warning_threshold': 80,  # Warning at 80% of threshold
                'critical_threshold': 90,  # Critical at 90% of threshold
                'advice': [
                    "Look for scholarships and grants",
                    "Consider online learning options",
                    "Buy used textbooks or rent them"
                ],
                'critical_advice': [
                    "Explore education financing options",
                    "Look for student discounts and deals",
                    "Consider part-time work or work-study programs"
                ]
            },
            'Utilities': {
                'threshold': 8000,  # Standard threshold for utilities
                'warning_threshold': 85,  # Warning at 85% of threshold
                'critical_threshold': 95,  # Critical at 95% of threshold
                'advice': [
                    "Implement energy-saving measures",
                    "Check for better utility plans",
                    "Fix any leaks or inefficiencies"
                ],
                'critical_advice': [
                    "Get an energy audit",
                    "Consider switching utility providers",
                    "Invest in energy-efficient appliances"
                ]
            }
        }
        
        # Get current month's spending for the category
        cursor.execute('''
            SELECT SUM(amount) as total_spent,
                   COUNT(*) as transaction_count,
                   MAX(amount) as highest_transaction
            FROM transactions
            WHERE user_id = ? 
            AND category = ?
            AND type = 'Expense'
            AND date >= date('now', 'start of month')
        ''', (session['user_id'], category))
        
        result = cursor.fetchone()
        current_spent = float(result['total_spent']) if result['total_spent'] else 0
        transaction_count = result['transaction_count']
        highest_transaction = float(result['highest_transaction']) if result['highest_transaction'] else 0
        
        # Get total income for the current month
        cursor.execute('''
            SELECT SUM(amount) as total_income
            FROM transactions
            WHERE user_id = ?
            AND type = 'Income'
            AND date >= date('now', 'start of month')
        ''', (session['user_id'],))
        
        income_result = cursor.fetchone()
        total_income = float(income_result['total_income']) if income_result['total_income'] else 0
        
        # Get historical data
        cursor.execute('''
            WITH monthly_spending AS (
                SELECT strftime('%Y-%m', date) as month,
                       SUM(amount) as monthly_amount,
                       COUNT(*) as monthly_count
                FROM transactions
                WHERE user_id = ? 
                AND category = ?
                AND type = 'Expense'
                GROUP BY strftime('%Y-%m', date)
            )
            SELECT 
                AVG(monthly_amount) as avg_spend,
                AVG(monthly_count) as avg_transactions,
                MAX(monthly_amount) as max_monthly_spend
            FROM monthly_spending
        ''', (session['user_id'], category))
        
        hist_result = cursor.fetchone()
        historical_avg = float(hist_result['avg_spend']) if hist_result['avg_spend'] else 0
        avg_transactions = float(hist_result['avg_transactions']) if hist_result['avg_transactions'] else 0
        max_monthly_spend = float(hist_result['max_monthly_spend']) if hist_result['max_monthly_spend'] else 0
        
        # Get category threshold and advice
        category_info = CATEGORY_THRESHOLDS.get(category, {
            'threshold': historical_avg * 1.2 if historical_avg > 0 else 5000,
            'warning_threshold': 75,
            'critical_threshold': 90,
            'advice': [
                "Track your spending in this category",
                "Set a budget and stick to it",
                "Look for ways to reduce unnecessary expenses"
            ],
            'critical_advice': [
                "Review all expenses in this category",
                "Create a detailed budget plan",
                "Consider temporary spending freeze"
            ]
        })
        
        threshold = category_info['threshold']
        advice_list = category_info['advice']
        critical_advice = category_info.get('critical_advice', advice_list)
        warning_threshold = category_info.get('warning_threshold', 75)
        critical_threshold = category_info.get('critical_threshold', 90)
        
        # Get category allocation percentage
        category_allocation = {
            'Housing': 0.30,      # 30% of income
            'Food': 0.15,         # 15% of income
            'Transportation': 0.15,# 15% of income
            'Utilities': 0.10,    # 10% of income
            'Healthcare': 0.10,   # 10% of income
            'Entertainment': 0.05, # 5% of income
            'Shopping': 0.05,     # 5% of income
            'Education': 0.10,    # 10% of income
            'Other': 0.05,  
            'Household' :0.01     # 5% of income
        }
        
        # Calculate category-specific monthly budget
        allocation_percentage = category_allocation.get(category, 0.006)  # Default to 5% if category not found
        monthly_budget = total_income * allocation_percentage if total_income > 0 else threshold
        
        # Generate response with enhanced analysis
        response = {
            'category': category,
            'current_spent': current_spent,
            'threshold': monthly_budget,  # Use monthly budget as threshold
            'historical_average': historical_avg,
            'total_income': total_income,
            'status': 'good',
            'advice': [],
            'transaction_stats': {
                'count': transaction_count,
                'highest_amount': highest_transaction,
                'avg_transactions_per_month': round(avg_transactions, 1),
                'highest_monthly_spend': max_monthly_spend
            }
        }
        
        # Calculate percentage based on monthly budget
        threshold_percentage = (current_spent / monthly_budget * 100) if monthly_budget > 0 else 0
        
        # Determine status and advice based on spending patterns
        if threshold_percentage >= critical_threshold:
            response['status'] = 'critical'
            response['message'] = f"⚠️ CRITICAL ALERT: You've spent {threshold_percentage:.1f}% of your {category} budget (₹{monthly_budget:,.2f})!"
            response['advice'] = critical_advice
            
            # Add specific recommendations based on transaction patterns
            if transaction_count > avg_transactions * 1.5:
                response['advice'].append(f"You've made {transaction_count} transactions this month, which is higher than your average of {avg_transactions:.1f}. Consider reducing transaction frequency.")
            
            if highest_transaction > monthly_budget * 0.5:
                response['advice'].append(f"Your highest transaction (₹{highest_transaction:.2f}) represents a significant portion of your budget. Try to avoid such large expenses.")
            
        elif threshold_percentage >= warning_threshold:
            response['status'] = 'warning'
            response['message'] = f"⚠️ Warning: You've spent {threshold_percentage:.1f}% of your {category} budget (₹{monthly_budget:,.2f})."
            response['advice'] = advice_list
            
            # Add preventive recommendations
            if current_spent > historical_avg:
                response['advice'].append(f"Your current spending (₹{current_spent:.2f}) is higher than your historical average (₹{historical_avg:.2f}). Consider reviewing your expenses.")
            
        elif threshold_percentage >= 50:
            response['status'] = 'notice'
            response['message'] = f"ℹ️ Notice: You've spent {threshold_percentage:.1f}% of your {category} budget (₹{monthly_budget:,.2f})."
            response['advice'] = [advice_list[0]]
            
            # Add planning recommendations
            if transaction_count < avg_transactions * 0.5:
                response['advice'].append("You still have room in your budget, but plan your remaining expenses carefully.")
            
        else:
            response['message'] = f"✅ Good job! You're well within your {category} budget (₹{monthly_budget:,.2f})."
            if current_spent < historical_avg * 0.5:
                response['advice'].append("You're spending less than usual this month - great work on budget management!")
        
        # Add trend analysis
        if historical_avg > 0:
            percent_diff = ((current_spent - historical_avg) / historical_avg * 100)
            if percent_diff > 20:
                response['trend'] = f"📈 Your spending is {percent_diff:.1f}% higher than your historical average."
                if percent_diff > 50:
                    response['trend'] += " This is a significant increase - consider reviewing your spending habits."
            elif percent_diff < -20:
                response['trend'] = f"📉 Your spending is {abs(percent_diff):.1f}% lower than your historical average."
                if current_spent < threshold * 0.3:
                    response['trend'] += " You're doing an excellent job managing expenses in this category!"
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error getting category advice: {e}")
        return jsonify({'error': 'Unable to generate advice at this time'}), 500
    finally:
        conn.close()

@app.route('/delete_transaction/<int:transaction_id>', methods=['DELETE'])
def delete_transaction(transaction_id):
    """Delete a specific transaction"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # First, verify that the transaction belongs to the current user
        cursor.execute('''
            SELECT id FROM transactions 
            WHERE id = ? AND user_id = ?
        ''', (transaction_id, session['user_id']))

        if not cursor.fetchone():
            return jsonify({'error': 'Transaction not found or unauthorized'}), 404

        # Delete the transaction
        cursor.execute('DELETE FROM transactions WHERE id = ?', (transaction_id,))
        conn.commit()

        return jsonify({'message': 'Transaction deleted successfully'})

    except Exception as e:
        app.logger.error(f"Error deleting transaction: {str(e)}")
        return jsonify({'error': 'Failed to delete transaction'}), 500

    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    app.run(debug=True)